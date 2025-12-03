from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
from src.replay import PrioritizedMemory
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from dataclasses import dataclass
from data import *
import math




"""

This projection layer takes the hidden state of the LLM as input and outputs a query to the AH.
This query is not english, but a learned representation for the LLM to query the AH. This query
is first received by the Entorhinal layer and proected into ent_dim.

Bascially, take 2048-D hidden state -> 512-D query vector
"""
class ProjectionHead(nn.Module):

    def __init__(self, input_dim=2048, output_dim=512):
        super().__init__()

        self.proj = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.proj_norm = nn.LayerNorm(normalized_shape=512, eps=1e-5)

    def forward(self, x):
        x = self.proj(x)
        x = self.proj_norm(x)
        return x


"""
The Entorhinal Layer

The Entorhinal Layer is a dense layer receiving the query/queue from the LLM that projects it
into a space that is understood and used by the AH. This layer doesn't need to be complex or fancy,
just a single nonlinear projection. This layer is preparing the query for pattern separation in the 
compression layer.

take 512-D -> 1024-D with some contextual embedding
"""
class EntorhinalLayer(nn.Module):
    def __init__(self, query_dim, ent_dim):
        super().__init__()

        self.project = nn.Linear(in_features=query_dim, out_features=ent_dim)
        self.nonlin = nn.SiLU()

    def forward(self, query):
        projection = self.project(query)
        projection = self.nonlin(projection)

        return projection
    

def kWTA(input, k):

    # Get the k-th largest value
    # topk returns values and indices, we need the k-th value
    # torch.topk returns values in descending order, so the k-th value is at index k-1
    kth_value = torch.topk(input, k, dim=-1)
    thresh = kth_value.values[:, -1].unsqueeze(-1)

    # Create a mask where values greater than or equal to the k-th value are True
    mask = (input >= thresh).to(input.dtype)

    # Apply the mask to the original tensor
    return input * mask

"""
The Compression Layer (Dentate Gyrus/CA3)

The compression layer is receives the output from the entorhinal layer and the storage retrieval score. It then signficiantly expands it into a very high dimension space that will be used to 
sparsify for pattern separation. The ReLU begins the pattern seprartion by removing negatives.
We significantly sparsify using K-Winner-Takes-All to finalize pattern separation. Our output should have ~2-5% activiation.

TL;DR: Separates patterns creating unique sparse code of memory.
"""

class CompressionLayer(nn.Module):
    def __init__(self, ent_dim, expansion, k=512):
        super().__init__()

        self.input_dim = ent_dim #WOULD ADD FAMILIARITY HERE
        self.k = k

        self.expand = nn.Linear(self.input_dim, ent_dim * expansion)
        self.nonlin = nn.ReLU()

    def forward(self, ent_output):
        
        expansion = self.expand(ent_output)
        #ReLU begin the pattern separation
        expansion = self.nonlin(expansion)

        #Sparsify our representation to complete pattern separation. 2-5% sparsity depending on k.
        z_sparse = kWTA(expansion, self.k)

        return z_sparse
        

"""
Stage 2 Memory (Abstract Reconstruction)

Our stage 2 memory is designed to extract a compressed, abstract gist of the memory. It does this by finding
the indices of active neurons in the sparse code, embedding the indices, attending to them with context, and 
producing a vector for each batch element.
"""

class Stage2(nn.Module):
    def __init__(self, z_dim, embd_dim):

        """
        Here we use a Perceiver Style Cross Attention mechanism. Our Stage 2 memory
        creates a more abstract representation of the memory being stored. This mechanism
        allows us to create num_latents "gist tokens" that each represent some part of the memory.
        The latent vectors are learned
        """
        super().__init__()

        self.embd = nn.Embedding(z_dim, embd_dim)

        self.attn_gate = nn.Sequential(
            nn.Linear(embd_dim, embd_dim),
            nn.Tanh(),
            nn.Linear(embd_dim, 1)
        )



    def forward(self, z_sparse, context_embedding):
        
        active_idx = (z_sparse > 0).nonzero(as_tuple=False) #Find the position of all nonzero elements in z_sparse 
        batch_idx = active_idx[:, 0] #Which batch each activation belongs to
        feature_idx = active_idx[:, 1] #Which feature is active

        emb = self.embd(feature_idx)
        ctx = context_embedding[batch_idx]

        attn_scores = (ctx * emb).sum(-1) / math.sqrt(emb.size(-1))
        attn_weights = torch.zeros_like(attn_scores)
        for b in range(z_sparse.size(0)):
            mask = (batch_idx == b)
            attn_weights[mask] = torch.softmax(attn_scores[mask], dim=0)

        out = torch.zeros(z_sparse.size(0), emb.size(1), device=z_sparse.device) #[B, embd_dim]

        #For each row index in batch_idx add emb[i] * attn_weights[i]
        out.index_add_(0, batch_idx, emb * attn_weights.unsqueeze(-1))

        #This is a vector where each element (counts[i]) is the frequency of i in batch_idx
        #Essentially, how many active features each batch sample has. Clamp ensures we don't
        # divide by zero if some batch sample had no active features
        counts = torch.bincount(batch_idx, minlength=z_sparse.size(0)).clamp(min=1)
        out /= counts.unsqueeze(-1)

        return out


"""
Stage 1 Memory (High-Fidelity Reconstuction)

Our Stage 1 memory is probably the most crucial part of this memory circuit. It is responsible for creating and 
maintaining a latent memory to avoid catastrophic forgetting and will be used for offline consolidation, it creates 
a context-rich cue for our Stage 2 abstraction, and computes the novelty of a memory. 
The most important of these roles is the latent memory. Catastrophic forgetting is an obivous hurdle for a module
such as this one, and creating an efficient, compact, and robust short-term memory was high priority. Our latent memory
is attended to by the entorhinal output to create a context-aware memory trace. THis trace won't be explicitly used 
for storage, but the trace will be replayed until we have a sufficiently high-fidelity reconstruction to avoid forgetting.
While computing the attention weights between the entorhinal output and latent memory, we are able to compute the 
novelty of the entorhinal output which is used to determine the strength with which we should "focus on" this memory.
More novel memories get more focus.
Finally, our Stage 1 memory receives the key and value vectors from the llm which contain rich context that we use 
to create a context embedding for our stage 2 memory.

TL;DR: Stage 1 is responsible for embedding a lot fo context from the entorhinal layer and LLM and maintain a 
short-term memory in the form of latent vectors to prevent catastropic forgetting.
"""
class Stage1(nn.Module):
    """
    Our Stage 1 will perform cross attention with the Key and Value from the attention block from
    the LLL and we will learn the query.
    """
    def __init__(self, ent_dim, proj_dim, hidden_dim, num_latents=64):
        super().__init__()
        self.alpha = 0.01
        self.num_latents = num_latents
        self.latent_memory = nn.Parameter(torch.zeros((num_latents,hidden_dim)))

        self.v_proj = nn.Linear(hidden_dim, proj_dim)
        self.k_proj = nn.Linear(hidden_dim, proj_dim)
        self.q_proj = nn.Linear(ent_dim, proj_dim)

        self.proj_dim = proj_dim

        self.llm_k_proj = nn.Linear(hidden_dim, proj_dim)
        self.llm_v_proj = nn.Linear(hidden_dim, proj_dim)

        self.ltm_gate = nn.Sequential(
            nn.Linear(num_latents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.mem_proj = nn.Linear(proj_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(proj_dim)
        self.llm_ln1 = nn.LayerNorm(proj_dim)
        
        self.novelty = None

        
    def _compute_novelty(self, weights):
        #High entropy = distributed attention = novel input
        """
        Weights are the sigmoid of our affinities and thus summing them
        along this dimension gives us the total familiarity or similarity to 
        previously learned keys of this query.

        """
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
        novelty = torch.sigmoid(entropy - torch.log(self.num_latents))
        return novelty
    

    def forward(self, ent_out, llm_k, llm_v):

        """
        We cross-attend between the latent memory vector and current memory query to extract useful memory information from the 
        latent memory.
        In a query pass, we perform a gated update to the latent memory which should perform a very slight update.
        """
        B, H, L, D_h = llm_k.shape #[batch, heads, query_len, head_dim]
        latent = self.latent_memory.unsqueeze(0).expand(B, -1, -1) #[B, num_latents, ent_dim]


        llm_k = llm_k.reshape(B, L, H * D_h)
        llm_v = llm_v.reshape(B, L, H * D_h) #[batch, query_len, hidden_dim]

        llm_k = self.llm_k_proj(llm_k) #[B, query_len, proj_dim]
        llm_v = self.llm_v_proj(llm_v)

        v = self.v_proj(latent) # [B, num_latents, proj_dim]
        k = self.k_proj(latent)
        q = self.q_proj(ent_out) #[B, proj_dim]

        affinities = torch.matmul(q.unsqueeze(1), k.transpose(-1, -2))/ math.sqrt(self.proj_dim) #[B, 1, num_latents]
        affinities = affinities.squeeze(1) #[B, num_latents]
        weights = torch.softmax(affinities, dim=-1)
        
        self.novelty = self._compute_novelty(weights)

        llm_aff = torch.matmul(q.unsqueeze(1), llm_k.transpose(-1, -2)) / math.sqrt(self.proj_dim) #[B, 1, query_len]
        llm_aff = llm_aff.squeeze(1) #[128,103]
        llm_weights = torch.softmax(llm_aff, dim=-1)

        context = torch.matmul(llm_weights.unsqueeze(1), llm_v).squeeze(1) #CHECK DIMS
        context = self.llm_ln1(context) #We can pass this to Stage 2 as context

        p_ltm = self.ltm_gate(weights).mean(dim=0)

        mem_output = torch.matmul(weights.unsqueeze(1), v).squeeze(1) #[B, hidden_dim]
        mem_output = self.ln1(mem_output)
        mem_output = self.mem_proj(mem_output)


        #Need to configure update of latent memory

        return mem_output, p_ltm, context
    
    def update(self, ent_out, mem_trace, novelty, reward, lr=0.05):
        """
        We want to update our latent memory to contain tractable information about the recent memory without
        forgetting significant detials of other stored memories.

        Our update should strongly embed novle and important memories, but very slowly override existing memories. 
        We want plasticity without forgetting.


        CURRENT SOLUTION:
        
        A Hebian-style gated update
        ent_out: [B, ent_dim]
        mem_trace: [B, hidden_dim]
        lr: learning rate or update strength

        POSSIBLE FUTURE EXTENSION:

        Take in feedback signal and do something like:
            novelty = torch.sigmoid(feedback_signal)
        if using a reward signal or something like that for dopamine-inspired modulation
        """

        q = self.q_proj(ent_out) #[B, proj_dim]
        k = self.k_proj(self.latent_memory) #[num_lantents, proj_dim]
        weights = torch.softmax(q @ k.T / math.sqrt(self.proj_dim), dim=-1) #[B, num_latents]

        candidate = self.v_proj(mem_trace)

        importance = novelty * torch.sigmoid(reward)

        momentum = 0.9
        update_stength = lr * importance.mean()

        for i in range(self.num_latents):
            slot_strength = weights[:, i].mean() * update_stength
            self.latent_memory.data[i] = (
                momentum * self.latent_memory.data[i] +
                slot_strength * candidate.mean(0) +
                (1 - momentum - slot_strength) * self.latent_memory.data[i]
            )

        #Homeostatic decay of unused slots (slowly forget old memories)
        decay_rate = 0.001
        unused_mask = (weights.mean(0) < 0.1)
        self.latent_memory.data[unused_mask] *= (1-decay_rate)


    
""" 
SAE (Neocortex)

The goal of this is to take our stage 2 vectors and create a reconstructible distributed representation of the memory.
Sparsity allows for greater pattern separation and thus a greater combination of unique memories that can be safely
stored.
"""

class SAE_LTM(nn.Module):
    def __init__(self, storage_dim, retrieval_dim, input_dim=512, hidden_dim=4096):
        super().__init__()
        
        self.storage_head = nn.Linear(storage_dim, input_dim)
        self.retrieval_head = nn.Linear(retrieval_dim, input_dim)
        self.silu = nn.SiLU()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.targets = None

        self.loss = nn.MSELoss()
        self.lr = 1e-5
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

    def forward(self, x, mode, k=512):

        if mode == 0:
            x = self.silu(self.storage_head(x))
        else:
            self.targets = x
            x = self.silu(self.retrieval_head(x))
            
        h = self.relu(self.encoder(x))

        #ADAPTIVE SPARISTY
        activation_entropy = -torch.sum(
            torch.softmax(h, dim=-1) * torch.log_softmax(h, dim=-1),
            dim=-1
        )
        #Higher entropy = less compressible = need more active units
        k = int(k * (1 + 0.5 * activation_entropy.mean() / torch.log(torch.tensor(h.size(-1)))))
        k = min(k, h.size(-1) // 2) #Cap sparsity at 50%

        z_sparse = kWTA(h, k)

        recon = self.decoder(z_sparse)
        return recon, z_sparse #We use this z_sparse for our loss, nothing else
    
    

    

"""
Integration Layer (CA1)

This layer does as it says; it integrates. It integrates our memory trace with context from various different parts of 
the memory module which have processed the query up to this point.
"""

class IntegrationLayer(nn.Module):
    def __init__(self, stm_dim, ent_dim, vae_dim):
        super().__init__()

        self.proj_stm = nn.Linear(stm_dim, ent_dim)
        self.proj_vae = nn.Linear(vae_dim, ent_dim)
        self.proj_ent = nn.Linear(ent_dim, ent_dim)

        self.gating = nn.Sigmoid()
        self.cross_attn = nn.MultiheadAttention(ent_dim, num_heads=4, batch_first=True)

        self.lin = nn.Linear(ent_dim * 2, ent_dim)


    def forward(self, stm, h_ent, vae, p_ltm):

        vae_proj = self.proj_vae(vae)
        stm_proj = self.proj_stm(stm)
        ent_proj = self.proj_ent(h_ent)

        ent_proj = ent_proj.unsqueeze(1)
        stm_proj = stm_proj.unsqueeze(1)
        vae_proj = vae_proj.unsqueeze(1)
        vae_proj = vae_proj * p_ltm
        """
        Concatenating along dimension 1 allow the attention mechanism to selectively
        use aspects of both z_proj and vae_proj in the key and value vectors.
        """
        memory = torch.cat([stm_proj, vae_proj], dim=1)
        int_mem, _ = self.cross_attn(ent_proj, memory, memory)
        int_mem = int_mem.squeeze(1)
        ent_proj = ent_proj.squeeze(1)
        gate = torch.cat([ent_proj, int_mem], dim=-1)
        g = self.gating(self.lin(gate))

        z_int = (1 - g) * ent_proj + g * int_mem

        return z_int

"""
Our output layer is the final projection from our integration layer back to our model's hidden state
dimension. This will then be passed into a Monosynaptic Injector (MSI) layers to be integrated with
the LLM's processing.
"""
class OutputLayer(nn.Module):
    def __init__(self, model_dim, int_dim):
        super().__init__()

        self.lin = nn.Linear(int_dim, model_dim) #model_dim is model embedding dim (2048 in tinyllama)
        self.nonlin = nn.SiLU()

    def forward(self, int_mem):

        mem = self.lin(int_mem)
        mem_final = self.nonlin(mem)

        return mem_final


"""
Monosynaptic Injection Layer (MSI Layer)

This layer is meant to mimic the brain's integration of memory signals from the hippocampus to 
the prefrontal cortex via monosynaptic connections. We use the hidden state of our LLM and the 
final memory output of our AH and gated addition with Feature-wise Linear Modulation (FiLM) to
control integration and allow the model to learn to utilize our memory's output.

We will use these layers to augment the hidden state of intermediate layers (19-22) in our LLM. This will allows the LLM to learn to use memory signals when it wants and process them alongside the rest
of the query context it has from previous layers.
"""

class MonosynapticInjector(nn.Module):
    def __init__(self, hidden_dim=2048, mem_dim=2048, use_film=True ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.mem_dim = mem_dim
        self.use_film = use_film

        self.mem_proj = nn.Sequential(
            nn.Linear(self.mem_dim, self.hidden_dim),
            nn.SiLU()
        )

        self.gate_dim = hidden_dim + mem_dim

        self.gate_net = nn.Sequential(
            nn.Linear(self.gate_dim, self.hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 4, 1)
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))
        """
        Feature-wise Linear Modulation will perform scaling and shifting of each feature, allowing
        the model to modulate each feature of the memory-injected hidden state.
        """
        self.film = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        )

        self.norm = nn.LayerNorm(self.hidden_dim)
    
    def forward(self, hidden_state, mem_final):

        B, L, H = hidden_state.shape
        mem_vec = self.mem_proj(mem_final)

        h_pool = hidden_state.mean(dim=1)
        gate_in = torch.cat([h_pool, mem_vec], dim=-1)

        gate_logits = self.gate_net(gate_in)
        gate = torch.sigmoid(gate_logits).view(B, 1, 1)

        mem_exp = mem_vec.unsqueeze(1)

        injected = hidden_state + self.alpha * gate * mem_exp

        if self.use_film:
            film_params = self.film(mem_final)
            gamma, beta = film_params.chunk(2, dim=-1)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            injected = gamma * injected + beta

        return self.norm(injected) 


class MemoryTrace(nn.Module):
    def __init__(self, prompt, completion, embd_dim, proj_dim, storage_dim, feedback=None, reward=None):
        super.__init__()
        
        #Ideally these are the tokens
        self.prompt = prompt
        self.completion = completion
        self.feedback = feedback
        self.reward = reward

        self.prompt_proj = nn.Linear(embd_dim, proj_dim)
        self.completion_proj = nn.Linear(embd_dim, proj_dim)
        self.feedback_proj = nn.Linear(embd_dim, proj_dim)

        self.trace_integrator = nn.Linear(proj_dim * 3, storage_dim)
        self.trace_act = nn.SiLU()

    def create_trace(self, llm):
        """
        Here we need to figure out what to do when we receive no explicit feedback from the user. Most of our training
        data will not include user feedback, we will likely need to create our own feedback examples
        """

        prompt_emb = llm.embed(self.prompt)
        completion_emb = llm.embed(self.completion)
        if self.feedback:
            feedback_emb = llm.embed(self.feedback)
            feedback_proj = self.feedback_proj(feedback_emb)
        else:
            feedback_proj = None

        prompt_proj = self.prompt_proj(prompt_emb)
        completion_proj = self.completion_proj(completion_emb)
        

        trace = torch.cat([prompt_proj, completion_proj, feedback_proj], dim=-1)
        integrated = self.trace_act(self.trace_integrator(trace))

        return integrated
        


        

@dataclass
class AH_Args:
    proj_dim = 512
    embd_dim = 2048
    ent_dim = 1024
    expansion = 10
    k = int(ent_dim * expansion * 0.05) #512 for ~5% sparsity
    z_dim = ent_dim * expansion
    stage_1_hidden = 256
    stage_2_context_dim = proj_dim #Our context is coming from our stage 1
    num_latents = 8
    stage_2_emb = proj_dim
    num_layers = 2
    storage_dim = stage_2_emb
    retrieval_dim = stage_2_emb
    vae_dim = 512
    int_dim = ent_dim
    batch_size = 128
    mem_dim = 256
    



class AH(nn.Module):
    def __init__(self, args:AH_Args):
        super().__init__()
        self.args = args

        self.stage1_loss = nn.MSELoss()
        self.ltm_thresh = 0.7

        self.entorhinal = EntorhinalLayer(args.proj_dim, args.ent_dim)
        self.compression = CompressionLayer(args.ent_dim, args.expansion, args.k)
        self.stage_1 = Stage1(args.ent_dim, args.proj_dim, args.stage_1_hidden)
        self.stage_2 = Stage2(args.z_dim, args.stage_2_emb)
        self.ltm = SAE_LTM(args.storage_dim, args.retrieval_dim)
        self.integration = IntegrationLayer(args.mem_dim, args.ent_dim, args.vae_dim)
        self.output = OutputLayer(args.embd_dim, args.int_dim)

        self.ent_out  = None #Will need to store for Stage 1 update later
        self.query = None #Will need for feedback module


        self.sae_input = None
        self.sae_recon = None
        
        self.stage_2_buffer = []

        self.final_output = None

    def forward(self, query, captured_kv):

        ent_out = self.entorhinal(query)
        z_sparse = self.compression(ent_out)

        
        self.stage1_input = z_sparse

        high_recon, context_embed = self.stage_1(z_sparse, captured_kv['k'], captured_kv['v']) 
        recon_loss = self.stage1_loss(query, high_recon)

        while recon_loss >= self.recon_thresh:
            self.stage1_optim.zero_grad()
            recon_loss.backward()
            self.stage1_optim.step()

            high_recon, context_embed = self.stage_1(z_sparse, captured_kv['k'], captured_kv['v']) 
            recon_loss = self.stage1_loss(query, high_recon)


        self.stage2_input = { 'z_sparse': z_sparse, 'context': context_embed} #store in some buffer until offline consolidation
        self.stage_2_buffer.append(self.stage2_input)

        if len(self.stage_2_buffer) > 100:
            #TIME FOR OFFLINE CONSOLIDATION
            #loop through and concat z_sparse and context for each stored memory

            #run each through stage 2
            for input in self.stage_2_buffer:
                latents = self.stage_2(input['z_sparse'], input['context']) #[Batch, num_latents, latent_size]
                latent_full = torch.stack(latents)
                self.stage2_recon = latent_full
                self.sae_input = latent_full
                sae_recon, loss_sparse = self.ltm(latent_full, 0)
                self.sae_recon = sae_recon


    def store(self, query, captured_kv, consolidation=False):
        
        ent_out = self.entorhinal(query)
        fam_score = self.familiarity_store(ent_out)
        z_sparse = self.compression(ent_out)

        
        self.stage1_input = z_sparse
        

        while recon_loss >= self.recon_thresh:
            self.stage1_optim.zero_grad()
            recon_loss.backward()
            self.stage1_optim.step()

            high_recon, context_embed = self.stage_1(z_sparse, captured_kv['k'], captured_kv['v']) 
            recon_loss = self.stage1_loss(query, high_recon)


        self.stage2_input = { 'z_sparse': z_sparse, 'context': context_embed} #store in some buffer until offline consolidation
        self.stage_2_buffer.append(self.stage2_input)

        if consolidation:
            #TIME FOR OFFLINE CONSOLIDATION
            #loop through and concat z_sparse and context for each stored memory

            #run each through stage 2
            for input in self.stage_2_buffer:
                latents = self.stage_2(input['z_sparse'], input['context']) #[Batch, num_latents, latent_size]
                latent_full = torch.stack(latents)
                self.stage2_recon = latent_full
                self.sae_input = latent_full
                sae_recon, loss_sparse = self.ltm(latent_full, 0)
                self.sae_recon = sae_recon

    def retrieve(self, query, captured_kv):

        self.query = query
        
        ent_out = self.entorhinal(query)
        self.ent_out = ent_out

        mem_trace, use_ltm, ctx = self.stage_1(ent_out, captured_kv['k'], captured_kv['v'])

        z_sparse = self.compression(ent_out)
        abstract = self.stage_2(z_sparse, ctx) #[B, proj_dim]
         #[B, num_latents * latent_dim]
        sae_out, z_sparse_loss = self.ltm(abstract, 1)
    
        int_out = self.integration(mem_trace, ent_out, sae_out, use_ltm)
        final = self.output(int_out)
        self.final_output = final

        return final


class LlamaWithAH(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        tokenizer,
        ah_module: AH,
        args: AH_Args,
        inject_layers,
        pause_layer: int = 15,
        device: torch.device = None,
        
    ):
        """
        pause_layer: layer index (1-based) at which we pause BEFORE executing that layer.
                     e.g., pause_layer=19 => run layers 0-18, then pause
        inject_layers: list of 0-based layer indices where MSI will be applied (after that layer's output)
        """
        super().__init__()
        
        self.device = device or torch.device("cpu")
        self.base = AutoModelForCausalLM.from_pretrained(base_model_name).to(self.device)
        self.tokenizer = tokenizer
        self.transformer = self.base.model  
        self.hidden_size = self.base.config.hidden_size
        self.args = args

        self.embeddings = None

        self.pause_layer = pause_layer
        self.inject_layers = inject_layers

        self.captured_kv = {}

        self.target_layer = self.transformer.layers[self.pause_layer - 1].self_attn
        self.hook_handle = self.target_layer.register_forward_hook(self.hook_func)

        self.query_attention = nn.MultiheadAttention(self.hidden_size, num_heads=8, batch_first=True)
        self.query_token = nn.Parameter(torch.randn(1,1,self.hidden_size))

        # Projection head (from last-token hidden state at pause -> AH query)
        self.projection_head = ProjectionHead(input_dim=self.hidden_size, output_dim=args.proj_dim).to(self.device)

        # AH module (user-provided) must implement retrieve_from_query(q) -> [B, hidden_size]
        self.ah = ah_module.to(self.device)

        # MSI modules for each injection layer
        self.msi_layers = nn.ModuleDict({
            str(l): MonosynapticInjector(hidden_dim=self.hidden_size, mem_dim=self.hidden_size).to(self.device)
            for l in self.inject_layers
        })

    def hook_func(self, module, input, output):
        hidden_states = output[0] #hidden_states: [1, 3, 2048]

        batch_size, sequence_len, _ = hidden_states.shape

        # project to K/V manually using module's weights
        k_proj = module.k_proj(hidden_states) #k_proj: [1, 3, 256] 
        v_proj = module.v_proj(hidden_states) #v_proj: [1, 3, 256]
        
        _, _, proj_dim = k_proj.shape
        """
        our embd_dim is 2048, our head dim is 64, and we have 32 heads, so we would expect this to be  64, but to speed up inference, they have the model have 4 key and value heads, with 32 query heads. So this results in our key and value projections being query_head_dim * 4, 256.
        As we would expect, when you look at the docs of the model, in the decoder layers, their key
        and query vectors are of length 256 and their query is 2048.
        """
        
        q_head_dim = module.head_dim #64 (embd_dim: 2048 / num_head: 32 = 64)
        num_kv_heads = proj_dim // q_head_dim


        k = k_proj.view(batch_size, sequence_len, num_kv_heads, q_head_dim).transpose(1, 2) #pre-transpose: [1, 3, 4, 64] post: [1, 4, 3, 64]
        v = v_proj.view(batch_size, sequence_len, num_kv_heads, q_head_dim).transpose(1, 2)
        #pre-transpose: [1, 3, 4, 64] post: [1, 4, 3, 64]

        self.captured_kv['k'] = k.detach()
        self.captured_kv['v'] = v.detach()

    def group_query(self, hidden_states):
        B, L, H = hidden_states.shape

        query = self.query_token.expand(B, -1, -1) #[Batch, 1, Hidden]

        attn_output, _ = self.query_attention(
            query = query,
            key = hidden_states,
            value = hidden_states
        )

        pooled = attn_output.squeeze(1)

        return pooled #[B, H]
        


    def forward(self, input_ids: torch.LongTensor, targets=None):
        """
        Runs: embed -> layers 0..pause_layer-1 -> pause & AH retrieval -> continue layers ->
        apply MSI at inject_layers -> final norm & lm_head -> logits
        """
        B, seq_len = input_ids.shape

        # embeddings (use underlying model's embedding)
        hidden_states = self.transformer.embed_tokens(input_ids).to(self.device)  # [B, L, H]
        self.embeddings = hidden_states

        B, L, H = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Create an attention mask of dimension [Batch, 1, L, L]
        # causal_mask = create_answer_only_mask(input_ids, tokenizer, device=device, dtype=dtype)
        attention_mask = create_answer_only_mask(input_ids, self.tokenizer, device, dtype)

        position_ids = torch.arange(0, L, device=device).unsqueeze(0).expand(B, L).long()  # [B, L]

        rotary_embd = self.transformer.rotary_emb
        position_embeddings = rotary_embd(hidden_states, position_ids)

        # iterate transformer layers manually
        num_layers = len(self.transformer.layers)

        mem_vec = None
              
        # run layers up to pause_layer-1 inclusive
        for i, layer in enumerate(self.transformer.layers):
            hidden_states = layer(hidden_states, 
                                 attention_mask=attention_mask,
                                 position_ids=position_ids,
                                 position_embeddings=position_embeddings
                                 )

            # hidden_states = layer_output[0]

            # if we've reached the layer before pause_layer, stop and retrieve
            if (i + 1) == self.pause_layer:
                query_pre = self.group_query(hidden_states)
                # project to AH query dim
                query = self.projection_head(query_pre)  # [B, q_dim]
                # Synchronous AH retrieval
                mem_vec = self.ah.retrieve(query, self.captured_kv)  # should be [B, H]
                # ensure mem_vec on same device and dtype
                mem_vec = mem_vec.to(device).type(dtype)
                # break out and continue from this exact hidden_states
                # note: do not apply MSI here
                break


        # Now continue remaining layers (from i+1 ... end)
        start_idx = i + 1
        for j in range(start_idx, num_layers):
            layer = self.transformer.layers[j]
            layer_outputs = layer(hidden_states, 
                                  attention_mask=attention_mask, 
                                  position_ids=position_ids,
                                  position_embeddings=position_embeddings)
            

            # apply MSI after this layer if configured
            if j in self.inject_layers:
                hidden_states = self.msi_layers[str(j)](hidden_states, mem_vec)

        # final layer norm & lm head (use base model's heads)
        hidden_states = self.transformer.norm(hidden_states)
        logits = self.base.lm_head(hidden_states)

        if targets:

            loss = F.cross_entropy(logits, targets, ignore_index=-100, reduction='mean')
        else:
            loss = 0

        return logits, loss

    # convenience method: decode via base tokenizer/generate without memory integration (use with care)
    def generate_with_memory(self, input_ids, max_new_tokens=256):
        # You need a custom generation loop to integrate AH at each step.
        # For now this delegates to base generate (no AH integration). Use only for testing.
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            print("PROBS: ",probs[:5])


            idx_next = probs.argmax(dim=-1, keepdim=True) #only take token with highest probability
            # append sampled index to the running sequence
            input_ids = torch.cat((input_ids, idx_next), dim=1) # (B, T+1)

        return input_ids
    
    def __del__(self):
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()


class RewardModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, query, response): #Receives the embeddings

        embs = torch.cat([query, response], dim=-1)

        attn_scores = self.attn(embs)
        weights = F.softmax(embs, dim=1)

        pooled = weights * embs
        pooled = F.normalize(pooled, dim=-1)

        r = self.net(pooled).squeeze(-1)

        return r
    

def jitter(memory, priority):
    sigma = 0.03 + 0.05 * priority

    memory = memory + torch.randn_like(memory) * sigma * memory.std(dim=0, keepdim=True)

    return memory


def store_memories(memories, rewards, novelties, buffer):
    reward_beta = 1e-5

    for memory, novelty, reward in zip(memories, novelties, rewards):
        priority = novelty * (1 + reward_beta * F.sigmoid(reward))
        buffer.add(memory, priority)



def consolidate(mem_buffer, sae:SAE_LTM, n_epochs):

    sae.optimizer.lr = sae.lr * 10 #boost lr for consolidation

    for epoch in range(n_epochs):

        batch, priorities = mem_buffer.get()

        batch_input = jitter(batch)
        recons, _ = sae(batch_input)

        for recon, target, priority in zip(recons, batch, priorities):
            loss = sae.loss(recon, target)

            weighted_loss = loss * priority
            weighted_loss.backward()

            sae.optimizer.step()
            sae.optimizer.zero_grad()

            

        


def stage_1_training():

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    tokenizer.padding_side = 'right'
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    args = AH_Args()
    ah = AH(args).to(device)

    model = LlamaWithAH(
        base_model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        tokenizer=tokenizer,
        ah_module=ah,
        args=args,
        inject_layers=[15,16,17,18],
        device=device
        ).to(device)
    
    # Freeze LLM weights
    for p in model.base.parameters():
        p.requires_grad = False

    # Collect trainable params (AH + MSI)
    trainable_params = list(model.ah.parameters()) + list(model.msi_layers.parameters()) + list(model.projection_head.parameters())

    optimizer = optim.AdamW(trainable_params, lr=1e-4)

    total_params = sum(p.numel() for p in trainable_params)

    trivia = load_dataset("trivia_qa", "rc.nocontext")
    trivia = load_dataset("trivia_qa", "rc.nocontext")  # "rc" = reading comprehension version


    # Access splits
    train_data = trivia["train"]

    train_data = train_data.remove_columns(['question_id', 'question_source', 'entity_pages', 'search_results'])
    questions = train_data['question']
    answers = train_data['answer']
    answers = [answer['value'] for answer in answers]

    separator = ' Answer:'
    dset = Stage1_Dataset(questions, answers, separator)
    dataloader = DataLoader(dset, batch_size=args.batch_size)

    mem_buffer = PrioritizedMemory(batch_size=args.batch_size)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
    consolidate_count = 0
    for batch in dataloader:
        
        # inputs = preprocess_fn(batch, tokenizer, max_len=256)
        inputs = tokenizer(batch, truncation=True, padding=True, return_tensors='pt', max_length=128)
        input_ids = inputs.input_ids.to(device)
        batch_size, seq_len = input_ids.shape
        targets = input_ids.clone()

        separator_ids = tokenizer.encode(separator, add_special_tokens=False)
    
        for b in range(args.batch_size):
            # Find where answer starts (after separator)
            answer_start = None
            for i in range(seq_len - len(separator_ids)):
                if input_ids[b, i:i+len(separator_ids)].tolist() == separator_ids:
                    answer_start = i + len(separator_ids)
                    break
            
            if answer_start is not None:
                # Mask prompt tokens (set to -100 so they're ignored by CrossEntropyLoss)
                targets[b, :answer_start] = -100
            else:
                # If separator not found, mask everything (safety)
                targets[b, :] = -100
            
        # Shift for next-token prediction
        targets = targets[:, 1:].contiguous()
        answer_mask = (targets != -100)
        # attn_mask = inputs.attention_mask.to(device)
        
        
        tokens, loss = model.forward(input_ids, targets) #GET TARGETS
        input_embds = model.embeddings
        memories = model.ah.ltm.targets
        novelties = model.ah.stage_1.novelty

        #Decode
        """
        We first need our token values of ONLY the completion and then get the embeddings of the completion.
        We pass the embeddings of our prompt and completion into our memory trace and reward model
        then we decode and pass to user (Or we could pass to user first and while they read we do the embeddings)
        """
        completions = tokenizer.batch_decode(tokens) 


        #Get Feedback 

        feedback = None #We need to get user response for the memory trace. Maybe make feedback optional
        """

        
        Thoughts on Feedback:

        When creating a memory, we need to know if what we said went over well to know if our memory is working
        well or not. An essential part of a memory is the result of our output. Having a feedback signal will
        help us get a gauge of where we are in terms of memory, reasoning, and the integration of these two.
        It will also let us know how "important" a memory is alongside its novelty as a memory that results
        in an extreme feedback signal either way should be remembered well.

        Currently, a reward model seems the most straight-forward method for getting a reward signal. An 
        important detail is that we will need to figure out how to interpret no direct feedback. 
        No direct feedback on a question is usually a good thing sincce this means that we answered the 
        question in its entirety and there is no need for further dialogue, but there may be instances
        where no feedback is a bad thing, so we will need to train our model on these scenarios.
        """
        reward_m = RewardModel(model.args.embd_dim, hidden_dim=512)
        rewards = reward_m(input_embds, completions)
        #Create memory trace
        mem_trace = MemoryTrace(input_ids, completions, model.args.embd_dim, model.args.storage_dim, feedback, rewards)
        mem_trace = mem_trace.create_trace(model)
        
        store_memories(memories, rewards, novelties, mem_buffer)
        consolidate_count += 1

        if consolidate_count >= 7:
            consolidate(mem_buffer, model.ah.ltm, n_epochs = 5)
            consolidate_count = 0
        
        model.ah.stage_1.update(model.ah.ent_out, mem_trace, novelties, rewards)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"loss: {loss.item():.4f}")


if __name__ == "__main__":
    stage_1_training()

"""
PROMPTS FOR TOMORROW:

1) For my reward model, my desire is for it to receive the prompt and completion embedding
and then output a scalar for the predicted reward. Thus, it would be trained with inputs of prompt and completion embeddings and a scalar target. However, it seems to be common practice 
for the reward model to learn to prefer certain types of responses and to output based on preference. Should I stick with my current plan to train a reward model to output a simple scalar or should I adopt a prefernce-style reward model?

2) Here is my current training plan:
Stage 1
I will first need to get a working reward model (hopefully we are able to find a pretrained one that outputs scalar rewards as desired). Once I have this, I will be freezeing the parameters of the language model and only training my AH parameters. I will be using the TriviaQA dataset for my stage 1 training and loss will be calculated on answers only. This will hopefully teach the AH to store facts and reintegrate them will the LLM.

Stage 2
In stage 2 I will unfreeze the LLM weights and begin training the whole system. I will be using the KILT dataset for stage 2. This should hopefully train the system to work as a whole and the AH will maintain its memory-storage role as it was taught in stage 1, and the LLM will learn to better utilize in Stage 2

Stage 3
Stage 3 will be RLHF, primarily for alignment and small semantic changes.

I'm concerned with my Stage 1 as I'm not sure this training method will work well with my model set up. 
The goal of my training is for the model to learn to work as a whole, letting the AH be responsible for memory storage and the LLM responsible for reasoning. 

Deeply analyze this train plan and provide strengths and weaknesses. If there are significant weaknesses or inefficiencies, provide a more optimal training plan.
"""