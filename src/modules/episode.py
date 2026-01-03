import torch
import torch.nn as nn

class MemoryTrace(nn.Module):
    def __init__(self, prompt, completion, embd_dim, proj_dim, storage_dim, feedback=None, reward=None):
        super.__init__()
        
        #Ideally these are the tokens
        self.prompt = prompt
        self.completion = completion
        self.feedback = feedback
        self.reward = reward

        self.prompt_proj = nn.Linear(embd_dim, proj_dim)
        self.completion_proj = nn.Linear(embd_dim, proj_dim)
        self.feedback_proj = nn.Linear(embd_dim, proj_dim)

        self.trace_integrator = nn.Linear(proj_dim * 3, storage_dim)
        self.trace_act = nn.SiLU()

    def create_trace(self, llm):
        """
        Here we need to figure out what to do when we receive no explicit feedback from the user. Most of our training
        data will not include user feedback, we will likely need to create our own feedback examples
        """

        prompt_emb = llm.embed(self.prompt)
        completion_emb = llm.embed(self.completion)
        if self.feedback:
            feedback_emb = llm.embed(self.feedback)
            feedback_proj = self.feedback_proj(feedback_emb)
        else:
            feedback_proj = None

        prompt_proj = self.prompt_proj(prompt_emb)
        completion_proj = self.completion_proj(completion_emb)
        

        trace = torch.cat([prompt_proj, completion_proj, feedback_proj], dim=-1)
        integrated = self.trace_act(self.trace_integrator(trace))

        return integrated
        

