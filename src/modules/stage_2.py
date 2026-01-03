import torch
import torch.nn as nn
import math

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
