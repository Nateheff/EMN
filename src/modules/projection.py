

"""

This projection layer takes the hidden state of the LLM as input and outputs a query to the AH.
This query is not english, but a learned representation for the LLM to query the AH. This query
is first received by the Entorhinal layer and proected into ent_dim.

Bascially, take 2048-D hidden state -> 512-D query vector
"""
class ProjectionHead(nn.Module):

    def __init__(self, input_dim=2048, output_dim=512):
        super().__init__()

        self.proj = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.proj_norm = nn.LayerNorm(normalized_shape=512, eps=1e-5)

    def forward(self, x):
        x = self.proj(x)
        x = self.proj_norm(x)
        return x
