import torch
import torch.nn as nn
import math

"""
Stage 1 Memory (High-Fidelity Reconstuction)

Our Stage 1 memory is probably the most crucial part of this memory circuit. It is responsible for creating and 
maintaining a latent memory to avoid catastrophic forgetting and will be used for offline consolidation, it creates 
a context-rich cue for our Stage 2 abstraction, and computes the novelty of a memory. 
The most important of these roles is the latent memory. Catastrophic forgetting is an obivous hurdle for a module
such as this one, and creating an efficient, compact, and robust short-term memory was high priority. Our latent memory
is attended to by the entorhinal output to create a context-aware memory trace. THis trace won't be explicitly used 
for storage, but the trace will be replayed until we have a sufficiently high-fidelity reconstruction to avoid forgetting.
While computing the attention weights between the entorhinal output and latent memory, we are able to compute the 
novelty of the entorhinal output which is used to determine the strength with which we should "focus on" this memory.
More novel memories get more focus.
Finally, our Stage 1 memory receives the key and value vectors from the llm which contain rich context that we use 
to create a context embedding for our stage 2 memory.

TL;DR: Stage 1 is responsible for embedding a lot fo context from the entorhinal layer and LLM and maintain a 
short-term memory in the form of latent vectors to prevent catastropic forgetting.
"""
class Stage1(nn.Module):
    """
    Our Stage 1 will perform cross attention with the Key and Value from the attention block from
    the LLL and we will learn the query.
    """
    def __init__(self, ent_dim, proj_dim, hidden_dim, num_latents=64):
        super().__init__()
        self.alpha = 0.01
        self.num_latents = num_latents
        self.latent_memory = nn.Parameter(torch.zeros((num_latents,hidden_dim)))

        self.v_proj = nn.Linear(hidden_dim, proj_dim)
        self.k_proj = nn.Linear(hidden_dim, proj_dim)
        self.q_proj = nn.Linear(ent_dim, proj_dim)

        self.proj_dim = proj_dim

        self.llm_k_proj = nn.Linear(hidden_dim, proj_dim)
        self.llm_v_proj = nn.Linear(hidden_dim, proj_dim)

        self.ltm_gate = nn.Sequential(
            nn.Linear(num_latents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.mem_proj = nn.Linear(proj_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(proj_dim)
        self.llm_ln1 = nn.LayerNorm(proj_dim)
        
        self.novelty = None

        
    def _compute_novelty(self, weights):
        #High entropy = distributed attention = novel input
        """
        Weights are the sigmoid of our affinities and thus summing them
        along this dimension gives us the total familiarity or similarity to 
        previously learned keys of this query.

        """
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
        novelty = torch.sigmoid(entropy - torch.log(self.num_latents))
        return novelty
    

    def forward(self, ent_out, llm_k, llm_v):

        """
        We cross-attend between the latent memory vector and current memory query to extract useful memory information from the 
        latent memory.
        In a query pass, we perform a gated update to the latent memory which should perform a very slight update.
        """
        B, H, L, D_h = llm_k.shape #[batch, heads, query_len, head_dim]
        latent = self.latent_memory.unsqueeze(0).expand(B, -1, -1) #[B, num_latents, ent_dim]


        llm_k = llm_k.reshape(B, L, H * D_h)
        llm_v = llm_v.reshape(B, L, H * D_h) #[batch, query_len, hidden_dim]

        llm_k = self.llm_k_proj(llm_k) #[B, query_len, proj_dim]
        llm_v = self.llm_v_proj(llm_v)

        v = self.v_proj(latent) # [B, num_latents, proj_dim]
        k = self.k_proj(latent)
        q = self.q_proj(ent_out) #[B, proj_dim]

        affinities = torch.matmul(q.unsqueeze(1), k.transpose(-1, -2))/ math.sqrt(self.proj_dim) #[B, 1, num_latents]
        affinities = affinities.squeeze(1) #[B, num_latents]
        weights = torch.softmax(affinities, dim=-1)
        
        self.novelty = self._compute_novelty(weights)

        llm_aff = torch.matmul(q.unsqueeze(1), llm_k.transpose(-1, -2)) / math.sqrt(self.proj_dim) #[B, 1, query_len]
        llm_aff = llm_aff.squeeze(1) #[128,103]
        llm_weights = torch.softmax(llm_aff, dim=-1)

        context = torch.matmul(llm_weights.unsqueeze(1), llm_v).squeeze(1) #CHECK DIMS
        context = self.llm_ln1(context) #We can pass this to Stage 2 as context

        p_ltm = self.ltm_gate(weights).mean(dim=0)

        mem_output = torch.matmul(weights.unsqueeze(1), v).squeeze(1) #[B, hidden_dim]
        mem_output = self.ln1(mem_output)
        mem_output = self.mem_proj(mem_output)


        #Need to configure update of latent memory

        return mem_output, p_ltm, context
    
    def update(self, ent_out, mem_trace, novelty, reward, lr=0.05):
        """
        We want to update our latent memory to contain tractable information about the recent memory without
        forgetting significant detials of other stored memories.

        Our update should strongly embed novle and important memories, but very slowly override existing memories. 
        We want plasticity without forgetting.


        CURRENT SOLUTION:
        
        A Hebian-style gated update
        ent_out: [B, ent_dim]
        mem_trace: [B, hidden_dim]
        lr: learning rate or update strength

        POSSIBLE FUTURE EXTENSION:

        Take in feedback signal and do something like:
            novelty = torch.sigmoid(feedback_signal)
        if using a reward signal or something like that for dopamine-inspired modulation
        """

        q = self.q_proj(ent_out) #[B, proj_dim]
        k = self.k_proj(self.latent_memory) #[num_lantents, proj_dim]
        weights = torch.softmax(q @ k.T / math.sqrt(self.proj_dim), dim=-1) #[B, num_latents]

        candidate = self.v_proj(mem_trace)

        importance = novelty * torch.sigmoid(reward)

        momentum = 0.9
        update_stength = lr * importance.mean()

        for i in range(self.num_latents):
            slot_strength = weights[:, i].mean() * update_stength
            self.latent_memory.data[i] = (
                momentum * self.latent_memory.data[i] +
                slot_strength * candidate.mean(0) +
                (1 - momentum - slot_strength) * self.latent_memory.data[i]
            )

        #Homeostatic decay of unused slots (slowly forget old memories)
        decay_rate = 0.001
        unused_mask = (weights.mean(0) < 0.1)
        self.latent_memory.data[unused_mask] *= (1-decay_rate)
