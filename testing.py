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


class EntorhinalLayer(nn.Module):
    def __init__(self, query_dim, ent_dim):
        super().__init__()

        self.project = nn.Linear(in_features=query_dim, out_features=ent_dim)
        self.nonlin = nn.SiLU()

    def forward(self, query):
        projection = self.project(query)
        projection = self.nonlin(projection)

        return projection

class Familiarity(nn.Module):
    def __init__(self, query_dim):
        super().__init__()

        self.lin = nn.Linear(query_dim, 1)

    def forward(self, query):
        return self.lin(query)
    

def kWTA(input, k):

    # Get the k-th largest value
    # topk returns values and indices, we need the k-th value
    # torch.topk returns values in descending order, so the k-th value is at index k-1
    kth_value = input.topk(k, largest=True, sorted=True).values[..., -1].unsqueeze(-1)

    # Create a mask where values greater than or equal to the k-th value are True
    mask = (input >= kth_value).to(input.dtype)

    # Apply the mask to the original tensor
    return input * mask

class CompressionLayer(nn.Module):
    def __init__(self, ent_dim, expansion, k=500):
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