import torch
import torch.nn as nn
from helpers import kWTA

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
        
