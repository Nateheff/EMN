import torch
import torch.nn as nn

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
