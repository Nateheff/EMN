from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


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
# inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
inputs = tokenizer(prompt, return_tensors='pt')

captured_kv = {}

def hook_func(module, input, output):
    hidden_states = output[0] #hidden_states: [1, 3, 2048]

    batch_size, sequence_len, _ = hidden_states.shape

    # project to K/V manually using module's weights
    k_proj = module.k_proj(hidden_states) #k_proj: [1, 3, 256] 
    v_proj = module.v_proj(hidden_states) #v_proj: [1, 3, 256]
    
    _, _, proj_dim = k_proj.shape
    """
    our embd_dim is 2048, our head dim is 64, and we have 32 heads, so we would expect this to be 
    64, but to speed up inference, they have the model have 4 key and value heads, with 32 query heads. So this results in our key and value projections being query_head_dim * 4, 256.
    """

    num_q_heads = module.num_heads #32
    
    q_head_dim = module.head_dim #64 (embd_dim: 2048 / num_head: 32 = 64)
    num_kv_heads = proj_dim // q_head_dim


    k = k_proj.view(batch_size, sequence_len, num_kv_heads, q_head_dim).transpose(1, 2) #pre-transpose: [1, 3, 4, 64] post: [1, 4, 3, 64]
    v = v_proj.view(batch_size, sequence_len, num_kv_heads, q_head_dim).transpose(1, 2)
    #pre-transpose: [1, 3, 4, 64] post: [1, 4, 3, 64]

    captured_kv['k'] = k.detach()
    captured_kv['v'] = v.detach()




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
    

class PFC(nn.Module):
    def __init__(self, base_model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"):
        super().__init__()

        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

        self.hidden_size = 2048 #Base model hidden size

        self.projection_head = ProjectionHead(input_dim=self.hidden_size, output_dim=256)

    def forward(self, input_ids, attention_mask=None, **kwargs):

        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True,  # We need hidden states for our projection
            output_attentions=True,
            **kwargs
        )

        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_size) [1, 3, 2048]

        projection = self.projection_head(last_hidden_state)

        return {
            'base_model_outputs': outputs,
            'projected_output': projection,
            'last_hidden_state': last_hidden_state
        }
    
    def generate(self, *args, **kwargs):
        #Delegate generation to the base model
        return self.base_model.generate(*args, **kwargs)


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

        cand = torch.cat(query, candidate)
        score = self.score(cand)
        return score

def kWTA(input, k):

    # Get the k-th largest value
    # topk returns values and indices, we need the k-th value
    # torch.topk returns values in descending order, so the k-th value is at index k-1
    kth_value = torch.topk(input, k, dim=-1)
    thresh = kth_value[:, -1].unsqueeze(-1)

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

        self.input_dim = ent_dim + 1 #Query + familiarity score
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
        attn_out = self.cross_attn(z_proj, llm_k, llm_v)
        z = self.ln1(z_proj + attn_out) #The vector we store as our Stage 2 context embedding
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

    def forward(self, x, mode):

        if mode == 0:
            x = self.silu(self.storage_head(x))
        else:
            x = self.silu(self.retrieval_head(x))
            
        h = self.relu(self.encoder(x))
        z_sparse = kWTA(h, 0.05)

        recon = self.decoder(z_sparse)
        return recon, z_sparse
    
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
        z_proj = self.proj_ent(z_sparse)
        ent_proj = self.proj_ent(h_ent)

        memory = torch.cat([z_sparse, vae_proj], dim=1)
        int_mem = self.cross_attn(ent_proj, memory, memory)

        gate = torch.cat([ent_proj, int_mem], dim=-1)
        g = self.gating(self.lin(gate))

        z_int = (1 - g) * ent_proj + g * int_mem

        return z_int

"""
NOTE: I think we should only add this to the LAST hidden state of the LLM as all the layers
up to that are just building context into that final hidden state and all we are doing is 
adding more context which is naturally cumulated in the final hidden state.
"""
class OutputLayer(nn.Module):
    def __init__(self, model_dim, int_dim):
        super().__init__()

        self.lin = nn.Linear(int_dim, model_dim) #model_dim is model embedding dim (2048 in tinyllama)
        self.lin_gate = nn.Linear(int_dim, 1)
        self.nonlin = nn.SiLU()

    def forward(self, int_mem):

        gate = self.lin_gate(int_mem)
        mem = self.lin(int_mem)
        mem_final = self.nonlin(mem)

        return mem_final, gate



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
    storage_dim = num_latents * latent_size + stage_2_context_dim
    retrieval_dim = z_dim
    vae_dim = 512
    int_dim = ent_dim
    



class EMN(nn.Module):
    def __init__(self, pfc, args:AH_Args):
        super().__init__()
        self.pfc = pfc
        self.args = args

        self.entorhinal = EntorhinalLayer(args.proj_dim, args.ent_dim)
        self.compression = CompressionLayer(args.ent_dim, args.expansion, args.k)
        self.stage_1 = Stage1(args.z_dim, args.embd_dim, args.stage_1_hidden)
        self.stage_2 = Stage2(args.z_dim, args.stage_2_context_dim, args.num_latents, args.latent_size, args.num_layers)
        self.ltm = SAE_LTM(args.storage_dim, args.retrieval_dim, args)
        self.integration = IntegrationLayer(args.z_dim, args.ent_dim, args.vae_dim)
        self.output = OutputLayer(args.embd_dim, args.int_dim)
        
        self.stage_2_buffer = []

    def storage(self, prompt):

        pfc_out = self.pfc(prompt)
        pfc_hidden_state = pfc_out['last_hidden_state']
        target_layer = model.base_model.base_model.layers[-1].self_attn
        hook_handle = target_layer.register_forward_hook(hook_func)

        ent_out = self.entorhinal(pfc_out['projected_output'])
        z_sparse = self.compression(ent_out)

        high_recon, context_embed = self.stage_1(z_sparse, captured_kv['k'], captured_kv['v']) 
        stage_2_input = { 'z_sparse': z_sparse, 'context': context_embed} #store in some buffer until offline consolidation
        self.stage_2_buffer.append(stage_2_input)

        #TIME FOR OFFLINE CONSOLIDATION
        #loop through and concat z_sparse and context for each stored memory

        #run each through stage 2
        for input in self.stage_2_buffer:
            latents = self.stage_2(input['z_sparse'], input['context']) #[Batch, num_latents, latent_size]
            latent_full = torch.stack(latents)
            memory_trace = torch.cat[latent_full, input['z_sparse']]
            sae_recon = self.ltm(memory_trace, 0)

    def retrieve(self, prompt):
        pfc_out = self.pfc(prompt)
        ent_out = self.entorhinal(pfc_out['projected_output'])
        z_sparse = self.compression(ent_out)

        sae_out = self.ltm(z_sparse, 1)
        int_out = self.integration(z_sparse, ent_out, sae_out)
        final, gate = self.output(int_out)

        #Do something with gate to check retrieval utility/confidence
        #add back to LLM hidden state



model = PFC()


model_outputs = model(inputs.input_ids)
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