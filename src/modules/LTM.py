import torch
import torch.nn as nn
from helpers import kWTA

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
    
    
