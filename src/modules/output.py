import torch
import torch.nn as nn

"""
Our output layer is the final projection from our integration layer back to our model's hidden state
dimension. This will then be passed into a Monosynaptic Injector (MSI) layers to be integrated with
the LLM's processing.
"""
class OutputLayer(nn.Module):
    def __init__(self, model_dim, int_dim):
        super().__init__()

        self.lin = nn.Linear(int_dim, model_dim) #model_dim is model embedding dim (2048 in tinyllama)
        self.nonlin = nn.SiLU()

    def forward(self, int_mem):

        mem = self.lin(int_mem)
        mem_final = self.nonlin(mem)

        return mem_final
