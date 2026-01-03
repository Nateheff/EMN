import torch
import torch.nn as nn

"""
The Entorhinal Layer

The Entorhinal Layer is a dense layer receiving the query/queue from the LLM that projects it
into a space that is understood and used by the AH. This layer doesn't need to be complex or fancy,
just a single nonlinear projection. This layer is preparing the query for pattern separation in the 
compression layer.

take 512-D -> 1024-D with some contextual embedding
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
    