from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn

"""
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
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

generate_ids = model.generate(inputs.input_ids, max_length=50)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output)
"""

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
            **kwargs
        )

        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_size)

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
    kth_value = input.topk(k, largest=True, sorted=True).values[..., -1].unsqueeze(-1)

    # Create a mask where values greater than or equal to the k-th value are True
    mask = (input >= kth_value).to(input.dtype)

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

        proj = self.input_proj(z_sparse).unsqueeze(1) # (B, 1, latent_dim) NOT CURRENTLY USING

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
        z = self.ln1(z_proj + attn_out)
        mlp_out = self.mlp(z)
        z_recon = self.ln2(z + mlp_out)

        return z_recon
    

"""
Integration Layer (CA1)

Our Integration Layer is responsible 
"""

class IntegrationLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

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