from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
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
This projection layer takes the lst hidden state of the LLM as input and outputs a query to the AH.
This query is not english, but a learned representation for the LLM to query the AH. This query
is first received by the Entorhinal layer and proected into ent_dim.
"""
class ProjectionHead(nn.Module):

    def __init__(self, input_dim=2048, output_dim=256):
        super().__init__()

        self.proj = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.proj_norm = nn.LayerNorm(normalized_shape=256, eps=1e-5)

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

Our Stage 2 memory creates an abstract reconstruction of the memory we are looking to store. This
is where our memory is contextualized and prepared for long-term storage. We replay through our stage
2 memory during downtime for long-term consolidation. We input not only z_sparse, but also the 
hidden state from our stage 1 memoy as this serves as a context embedding.
"""

class Stage2(nn.Module):
    def __init__(self, z_dim, ctx_dim, num_latents, latent_size, num_layers):

        """
        Here we use a Perceiver Style Cross Attention mechanism. Our Stage 2 memory
        creates a more abstract representation of the memory being stored. This mechanism
        allows us to create num_latents "gist tokens" that each represent some part of the memory.
        The latent vectors are learned
        """
        super().__init__()

        self.input_proj = nn.Linear(z_dim + ctx_dim, latent_size)

        self.latents = nn.Parameter(torch.randn((num_latents, latent_size)))

        self.cross_attention = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model = latent_size,
                nhead=4,
                dim_feedforward=latent_size * 2,
                batch_first=True
            )

            for _ in range(num_layers)
        ])

    def forward(self, z_sparse, context_embedding):
        # Expand latents for batch
        B = z_sparse.size(0)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        contextualized = torch.cat([z_sparse, context_embedding], dim=-1)
        proj = self.input_proj(contextualized).unsqueeze(1) # (B, 1, latent_dim) NOT CURRENTLY USING

        for layer in self.cross_attention:
            latents = layer(tgt=latents, memory=proj) #latents are used for query, proj for key and value

        return latents #(B, num_latents, latent_size)


"""
Stage 1 Memory (High-Fidelity Reconstuction)

Our Stage 1 Memory is responsible for avoid catastrophic forgetting. The layer takes in just z_sparse as input, but also uses the key and value vectors from our LLM to contextualize the reconstruction. This layer's output is never actually used, but this is intentional. What we want from this layer is context. We get context in two forms: gradient and hidden state. Each memory that is stored is first passed through Stage 1 and ran through multiple times with an update until we have an accurate enough reconstruction (determined by familiarity score). These updates create a very rich hidden state that is then passed as context into our Stage 2 memory when it is time for long-term consolidation. The gradients from these updates flow to the earlier layers of the AH ensuring that the mappings and weights for this memory are persisted. Loss is computed by comparing reconstruction to original LLM embedding for that memory.
"""
class Stage1(nn.Module):
    """
    Our Stage 1 will perform cross attention with the Key and Value from the attention block from
    the LLL and we will learn the query.
    """
    def __init__(self, ent_dim, proj_dim, hidden_dim):
        super().__init__()

        self.latent_memory = nn.Parameter(torch.zeros((1,hidden_dim)))

        self.v_proj = nn.Linear(hidden_dim, proj_dim)
        self.k_proj = nn.Linear(hidden_dim, proj_dim)

        self.q_proj = nn.Linear(ent_dim, proj_dim)
        self.proj_dim = proj_dim

        self.ltm_gate = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.lin_gate = nn.Linear(proj_dim, hidden_dim)
        self.gating = nn.Sigmoid()
        
        self.ln1 = nn.LayerNorm(proj_dim)
        

    def forward(self, ent_out):

        """
        We cross-attend between the latent memory vector and current memory query to extract useful memory information from the 
        latent memory.
        In a query pass, we perform a gated update to the latent memory which should perform a very slight update.
        """

        v = self.v_proj(self.latent_memory)
        k = self.k_proj(self.latent_memory)

        q = self.q_proj(ent_out)

        affinities = q @ k.T / math.sqrt(self.proj_dim) #USE THIS TO DECIDE LTM OR NOT

        need_ltm = self.gating(self.ltm_gate(affinities.mean(dim=0)))

        mem_output = self.ln1(affinities @ v) #CHECK DIMS

        #GATED UPDATE OF LATENT MEMORY
        gates = self.gating(self.lin_gate(mem_output))
        self.latent_memory = (1 - gates.mean(0)) * self.latent_memory + gates.mean(0) * mem_output.mean(0)

        return mem_output, need_ltm
    
""" 
Hypernetwork to transition latent-stored memories to LTM? 
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

    def forward(self, x, mode, k=512):

        if mode == 0:
            x = self.silu(self.storage_head(x))
        else:
            x = self.silu(self.retrieval_head(x))
            
        h = self.relu(self.encoder(x))
        z_sparse = kWTA(h, k)

        recon = self.decoder(z_sparse)
        return recon, z_sparse #We use this z_sparse for our loss, nothing else
    
    def loss(self, recon, target_1, target_2, z_sparse, beta=0.3, gamma=1e-4):  #Maybe add importance weighting

        
        stage_1_recon = F.mse_loss(recon, target_1)

        stage_2_recon = F.mse_loss(recon, target_2)

        sparsity_loss = torch.mean(torch.abs(z_sparse))

        return stage_1_recon + (beta * stage_2_recon) + (gamma * sparsity_loss)
    

    

"""
Integration Layer (CA1)

Our Integration Layer is responsible for integrating memory traces with context. From the Compression layer (CA3) the Intregation layer receives the top z_sparse traces stores in our memory buffer and then integrates them with context from the entorhinal layer.
From the VAE LTM, the Integration layer recevies the decoder reconstructions and integrates them with context from the entorhinal layer
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


    def forward(self, stm, h_ent, vae):

        vae_proj = self.proj_vae(vae)
        stm_proj = self.proj_stm(stm)
        ent_proj = self.proj_ent(h_ent)

        ent_proj = ent_proj.unsqueeze(1)
        stm_proj = stm_proj.unsqueeze(1)
        vae_proj = vae_proj.unsqueeze(1)

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


@dataclass
class AH_Args:
    proj_dim = 512
    embd_dim = 2048
    ent_dim = 1024
    expansion = 10
    k = int(ent_dim * expansion * 0.05) #512 for ~5% sparsity
    z_dim = ent_dim * expansion
    stage_1_hidden = 1024
    stage_2_context_dim = embd_dim
    num_latents = 8
    latent_size = 256
    num_layers = 2
    storage_dim = num_latents * latent_size
    retrieval_dim = z_dim
    vae_dim = 512
    int_dim = ent_dim
    



class AH(nn.Module):
    def __init__(self, args:AH_Args):
        super().__init__()
        self.args = args

        self.stage1_loss = nn.MSELoss()
        self.recon_thresh = 10

        self.entorhinal = EntorhinalLayer(args.proj_dim, args.ent_dim)
        self.compression = CompressionLayer(args.ent_dim, args.expansion, args.k)
        self.stage_1 = Stage1(args.ent_dim, args.proj_dim, args.stage_1_hidden)
        self.stage_2 = Stage2(args.z_dim, args.stage_2_context_dim, args.num_latents, args.latent_size, args.num_layers)
        self.ltm = SAE_LTM(args.storage_dim, args.retrieval_dim)
        self.integration = IntegrationLayer(args.z_dim, args.ent_dim, args.vae_dim)
        self.output = OutputLayer(args.embd_dim, args.int_dim)



        #NOTE: We have these spread throughout, and they are (for the most part) WRONG! When we need 
        # to be calculating these recon losses online for stage 1 and SAE. 


        self.sae_input = None
        self.sae_recon = None
        
        self.stage_2_buffer = []

        self.final_output = None

    def froward(self, query, captured_kv):

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


    def storage(self, query, captured_kv, consolidation=False):
        
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

    def retrieve(self, query):
        
        ent_out = self.entorhinal(query)
        mem_trace, use_ltm = self.stage_1(ent_out)

        if use_ltm:
            z_sparse = self.compression(ent_out)
            latents = self.stage_2(z_sparse)
            latent_full = torch.stack(latents)
            sae_out = self.ltm(latent_full, 1)
        
        int_out = self.integration(mem_trace, ent_out, sae_out)
        final = self.output(int_out)
        self.final_output = final

        return final

