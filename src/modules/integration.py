import torch
import torch.nn as nn

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
