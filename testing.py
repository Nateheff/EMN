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
    

"""
Familiarity Score

Our familiary score is a learned scalar that represents how novel a memory is. If a memory is novel
or receives a higher familiarity score, that memory will be given higher priority for replay. This
score will be used for many purposes:
First (Storage), this score will be used as a priority metric allowing the memory to optimize storage.
Second (Retrieval), it serves as a threshold for retrieved memories. When we retrieve a memory, we pass it back through our familiarity layer and if the score is not beyond a threshold, we do not use it.

"""
class FamiliarityStorage(nn.Module):
    def __init__(self, query_dim):
        super().__init__()

        self.lin = nn.Linear(query_dim, 1)

    def forward(self, query):
        return self.lin(query)
    

class FamiliarityRetrieval(nn.Module):
    def __init__(self, query_dim, candidate_dim):
        super().__init__()

        self.score = nn.Linear(query_dim + candidate_dim, 1)

    def forward(self, query, candidate):

        cand = torch.cat([query, candidate], dim=-1)
        score = self.score(cand)
        return score

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
    def __init__(self, z_dim, emb_dim, hidden_dim):
        super().__init__()

        self.input_proj = nn.Linear(z_dim, emb_dim)
        self.cross_attn = nn.MultiheadAttention(emb_dim, num_heads=4, batch_first=True)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, emb_dim)
        )
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, z_sparse, llm_k, llm_v):

        z_proj = self.input_proj(z_sparse)
        attn_out, _ = self.cross_attn(z_proj, llm_k, llm_v)
        z = self.ln1(z_proj + attn_out) #The vector we store as our Stage 2 context embedding (Might need attn_out[0])
        mlp_out = self.mlp(z)
        z_recon = self.ln2(z + mlp_out)

        return z_recon, z
    


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
    def __init__(self, z_dim, ent_dim, vae_dim):
        super().__init__()

        self.proj_z = nn.Linear(z_dim, ent_dim)
        self.proj_vae = nn.Linear(vae_dim, ent_dim)
        self.proj_ent = nn.Linear(ent_dim, ent_dim)

        self.gating = nn.Sigmoid()
        self.cross_attn = nn.MultiheadAttention(ent_dim, num_heads=4, batch_first=True)

        self.lin = nn.Linear(ent_dim * 2, ent_dim)


    def forward(self, z_sparse, h_ent, vae):

        vae_proj = self.proj_vae(vae)
        z_proj = self.proj_z(z_sparse)
        ent_proj = self.proj_ent(h_ent)

        ent_proj = ent_proj.unsqueeze(1)
        z_proj = z_proj.unsqueeze(1)
        vae_proj = vae_proj.unsqueeze(1)

        """
        Concatenating along dimension 1 allow the attention mechanism to selectively
        use aspects of both z_proj and vae_proj in the key and value vectors.
        """
        memory = torch.cat([z_proj, vae_proj], dim=1)
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
    proj_dim = 256
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

        self.familiarity_store = FamiliarityStorage(args.proj_dim)

        self.entorhinal = EntorhinalLayer(args.proj_dim, args.ent_dim)
        self.compression = CompressionLayer(args.ent_dim, args.expansion, args.k)
        self.stage_1 = Stage1(args.z_dim, args.embd_dim, args.stage_1_hidden)
        self.stage_2 = Stage2(args.z_dim, args.stage_2_context_dim, args.num_latents, args.latent_size, args.num_layers)
        self.ltm = SAE_LTM(args.storage_dim, args.retrieval_dim)
        self.integration = IntegrationLayer(args.z_dim, args.ent_dim, args.vae_dim)
        self.output = OutputLayer(args.embd_dim, args.int_dim)
        


        #NOTE: We have these spread throughout, and they are (for the most part) WRONG! When we need 
        # to be calculating these recon losses online for stage 1 and SAE. 



        self.stage1_input = None
        self.stage1_recon = None

        self.stage2_input = None
        self.stage2_recon = None

        self.sae_input = None
        self.sae_recon = None
        
        self.stage_2_buffer = []

        self.final_output = None

    def storage(self, query, captured_kv, consolidation=False):
        
        ent_out = self.entorhinal(query)
        z_sparse = self.compression(ent_out)
        
        self.stage1_input = z_sparse
        high_recon, context_embed = self.stage_1(z_sparse, captured_kv['k'], captured_kv['v']) 
        self.stage1_recon = high_recon

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
        z_sparse = self.compression(ent_out)

        sae_out, loss_sparse = self.ltm(z_sparse, 1)
        int_out = self.integration(z_sparse, ent_out, sae_out)
        final = self.output(int_out)
        self.final_output = final

        return final


class LlamaWithAH(nn.Module):
    def __init__(
        self,
        base_model_name: str,
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
        self.transformer = self.base.model  
        self.hidden_size = self.base.config.hidden_size

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

        num_q_heads = module.num_heads #32
        
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
        


    def forward(self, input_ids: torch.LongTensor):
        """
        Runs: embed -> layers 0..pause_layer-1 -> pause & AH retrieval -> continue layers ->
        apply MSI at inject_layers -> final norm & lm_head -> logits
        """
        B, seq_len = input_ids.shape

        # embeddings (use underlying model's embedding)
        hidden_states = self.transformer.embed_tokens(input_ids).to(self.device)  # [B, L, H]

        B, L, H = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Create an attention mask of dimension [Batch, 1, L, L]
        causal_mask = _create_4d_causal_attention_mask(input_shape=[B,L], device=device, dtype=dtype)
        
        position_ids = torch.arange(0, L, device=device).unsqueeze(0).expand(B, L).long()  # [B, L]
        # iterate transformer layers manually
        num_layers = len(self.transformer.layers)

        mem_vec = None
              
        # run layers up to pause_layer-1 inclusive
        for i, layer in enumerate(self.transformer.layers):
            
            layer_output = layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids)
            hidden_states = layer_output[0]

            # if we've reached the layer before pause_layer, stop and retrieve
            if (i + 1) == self.pause_layer:
                query_pre = self.group_query(hidden_states)
                # project to AH query dim
                query = self.projection_head(query_pre)  # [B, q_dim]
                # Synchronous AH retrieval
                mem_vec = self.ah.retrieve(query)  # should be [B, H]
                # ensure mem_vec on same device and dtype
                mem_vec = mem_vec.to(device).type(dtype)
                # break out and continue from this exact hidden_states
                # note: do not apply MSI here
                break


        # Now continue remaining layers (from i+1 ... end)
        start_idx = i + 1
        for j in range(start_idx, num_layers):
            layer = self.transformer.layers[j]
            layer_outputs = layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids)
            hidden_states = layer_outputs[0]

            # apply MSI after this layer if configured
            if j in self.inject_layers:
                hidden_states = self.msi_layers[str(j)](hidden_states, mem_vec)

        # final layer norm & lm head (use base model's heads)
        hidden_states = self.transformer.norm(hidden_states)
        logits = self.base.lm_head(hidden_states)

        return logits

    # convenience method: decode via base tokenizer/generate without memory integration (use with care)
    def generate_with_memory(self, input_ids, max_new_tokens=256):
        # You need a custom generation loop to integrate AH at each step.
        # For now this delegates to base generate (no AH integration). Use only for testing.
        for _ in range(max_new_tokens):
            logits = self(input_ids)
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



class AHLoss(nn.Module):
    def __init__(self, lambda_stage1=1.0, lambda_stage2=1.0,
                 lambda_sae=1.0, lambda_ah=1.0, lambda_lm=0.0):
        super().__init__()
        self.l1 = lambda_stage1
        self.l2 = lambda_stage2
        self.ls = lambda_sae
        self.la = lambda_ah
        self.ll = lambda_lm

        self.mse = nn.MSELoss()

    def forward(self, stage1_in, stage1_recon, stage2_in, stage2_recon, sae_in, sae_recon, ah_output, stage1_norm, stage2_norm, sae_norm, ah_target=None, lm_logits=None, lm_labels=None):

        loss = 0.0

        if stage1_in is not None and stage1_recon is not None:
            loss += self.l1 * self.mse(stage1_recon, stage1_in) * stage1_norm
        
        if stage2_in is not None and stage2_recon is not None:
            loss += self.l2 * self.mse(stage2_recon, stage2_in) * stage2_norm

        if sae_in is not None and sae_recon is not None:
            loss += self.ls * self.mse(sae_recon, sae_in) * sae_norm

        # AH/global loss
        if ah_target is not None:
            loss += self.la * self.mse(ah_output, ah_target)

        # optional LLM cross-entropy
        if lm_logits is not None and lm_labels is not None:
            ce_loss = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                lm_labels.view(-1),
                ignore_index=-100
            )
            loss += self.ll * ce_loss

        return loss
        

def stage_1_training():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = AH_Args()
    ah = AH(args).to(device)

    model = LlamaWithAH(
        base_model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
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

    loss_fn = AHLoss(
    lambda_stage1=1.0,
    lambda_stage2=1.0,
    lambda_sae=1.0,
    lambda_ah=0.5,
    lambda_lm=0.1  # small weight on LM loss early
    )
    """
    FIX (Count correctly: numel())
    total_param = len(trainable_params)
    stage1_norm = len(model.ah.stage_1.parameters()) / total_param
    stage2_norm = len(model.ah.stage_2.parameters()) / total_param
    sae_norm = len(model.ah.ltm.parameters()) / total_param
    """
    trivia = load_dataset("trivia_qa", "rc.nocontext")
    dataloader = DataLoader(trivia['train'], batch_size=128)
    
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    tokenizer.padding_side = 'right'
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
    
    for batch in dataloader:
        inputs = preprocess_fn(batch, tokenizer, max_len=256)

        inputs = tokenizer(inputs, return_tensors='pt')
        input_ids = inputs.input_ids.to(device)
        attn_mask = inputs.attention_mask.to(device)

        args = AH_Args()
        ah = AH(args).to(device)

        model = LlamaWithAH(
            base_model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            ah_module=ah,
            args=args,
            inject_layers=[15,16,17,18],
            device=device
            ).to(device)
        
        logits = model.forward(input_ids)

        loss = loss_fn(
            model.ah.stage1_input,
            model.ah.stage1_recon,
            model.ah.stage2_input,
            model.ah.stage2_recon,
            model.ah.sae_input,
            model.ah.sae_recon,
            model.ah.final_output,
            stage1_norm,
            stage2_norm,
            sae_norm,
            ah_target,
            logits, 
            targets
            )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"loss: {loss.item():.4f}")
    """
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward pass
        outputs = model.forward(input_ids, attention_mask=attention_mask)

        # Unpack intermediate outputs from your AH
        stage1_in, stage1_recon = model.ah.stage1_inputs, model.ah.stage1_recons
        stage2_in, stage2_recon = model.ah.stage2_inputs, model.ah.stage2_recons
        sae_in, sae_recon = model.ah.sae_inputs, model.ah.sae_recons
        ah_out = model.ah.final_output

        # Logits from LLM (already returned)
        lm_logits = outputs
        lm_labels = labels

        # Compute total loss
        loss = loss_fn(
            stage1_in, stage1_recon,
            stage2_in, stage2_recon,
            sae_in, sae_recon,
            ah_out, ah_target=None,  # optional
            lm_logits=lm_logits, lm_labels=lm_labels
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"loss: {loss.item():.4f}")
"""

def test():
    
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    # model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    tokenizer.padding_side = 'right'
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    prompts = [
        "Hello, can you talk to me?",
        "What is the capital of France?",
        "I'm sad",
        "What are the seven days of the week?",
        "Hello",
        "I really like ice cream"
        ]

    prompt = "Hello."
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs.input_ids.to(device)
    attn_mask = inputs.attention_mask.to(device)

    args = AH_Args()
    ah = AH(args).to(device)

    model = LlamaWithAH(
        base_model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        ah_module=ah,
        args=args,
        inject_layers=[15,16,17,18],
        device=device
        ).to(device)
    
    generation = model.generate_with_memory(input_ids, max_new_tokens=20)

    decode = tokenizer.batch_decode(generation, skip_special_tokens=True)
    print(decode)



if __name__ == "__main__":
    stage_1_training()

# output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
# print(output)


"""
if __name__ == "__main__":
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    tokenizer.padding_side = 'right'
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    
    # Create extended model
    extended_model = PFC()
    
    # Test with your prompt
    prompt = "Hello."
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # Forward pass through extended model
    with torch.no_grad():
        outputs = extended_model(**inputs)
        
    print("Projected output shape:", outputs['projected_output'].shape)
    print("Last hidden state shape:", outputs['last_hidden_state'].shape)
    
    # You can still use generation from the base model
    generate_ids = extended_model.generate(inputs.input_ids, max_length=50)
    output_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    print(output_text)

"""