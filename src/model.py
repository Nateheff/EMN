from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
from src.replay import PrioritizedMemory
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset

from data import *
import math

from modules.projection import ProjectionHead
from modules.monosynaptic import MonosynapticInjector
from modules.episode import MemoryTrace
from AH import *


class LlamaWithAH(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        tokenizer,
        ah_module: AH,
        args: AH_Args,
        inject_layers,
        pause_layer: int = 15,
        device: torch.device = None,
        
    ):
        """
        pause_layer: layer index (1-based) at which we pause BEFORE executing that layer.
                     e.g., pause_layer=19 => run layers 0-18, then pause
        inject_layers: list of 0-based layer indices where MSI will be applied (after that layer's output)
        """
        super().__init__()
        
        self.device = device or torch.device("cpu")
        self.base = AutoModelForCausalLM.from_pretrained(base_model_name).to(self.device)
        self.tokenizer = tokenizer
        self.transformer = self.base.model  
        self.hidden_size = self.base.config.hidden_size
        self.args = args

        self.embeddings = None

        self.pause_layer = pause_layer
        self.inject_layers = inject_layers

        self.captured_kv = {}

        self.target_layer = self.transformer.layers[self.pause_layer - 1].self_attn
        self.hook_handle = self.target_layer.register_forward_hook(self.hook_func)

        self.query_attention = nn.MultiheadAttention(self.hidden_size, num_heads=8, batch_first=True)
        self.query_token = nn.Parameter(torch.randn(1,1,self.hidden_size))

        # Projection head (from last-token hidden state at pause -> AH query)
        self.projection_head = ProjectionHead(input_dim=self.hidden_size, output_dim=args.proj_dim).to(self.device)

        # AH module (user-provided) must implement retrieve_from_query(q) -> [B, hidden_size]
        self.ah = ah_module.to(self.device)

        # MSI modules for each injection layer
        self.msi_layers = nn.ModuleDict({
            str(l): MonosynapticInjector(hidden_dim=self.hidden_size, mem_dim=self.hidden_size).to(self.device)
            for l in self.inject_layers
        })

    def hook_func(self, module, input, output):
        hidden_states = output[0] #hidden_states: [1, 3, 2048]

        batch_size, sequence_len, _ = hidden_states.shape

        # project to K/V manually using module's weights
        k_proj = module.k_proj(hidden_states) #k_proj: [1, 3, 256] 
        v_proj = module.v_proj(hidden_states) #v_proj: [1, 3, 256]
        
        _, _, proj_dim = k_proj.shape
        """
        our embd_dim is 2048, our head dim is 64, and we have 32 heads, so we would expect this to be  64, but to speed up inference, they have the model have 4 key and value heads, with 32 query heads. So this results in our key and value projections being query_head_dim * 4, 256.
        As we would expect, when you look at the docs of the model, in the decoder layers, their key
        and query vectors are of length 256 and their query is 2048.
        """
        
        q_head_dim = module.head_dim #64 (embd_dim: 2048 / num_head: 32 = 64)
        num_kv_heads = proj_dim // q_head_dim


        k = k_proj.view(batch_size, sequence_len, num_kv_heads, q_head_dim).transpose(1, 2) #pre-transpose: [1, 3, 4, 64] post: [1, 4, 3, 64]
        v = v_proj.view(batch_size, sequence_len, num_kv_heads, q_head_dim).transpose(1, 2)
        #pre-transpose: [1, 3, 4, 64] post: [1, 4, 3, 64]

        self.captured_kv['k'] = k.detach()
        self.captured_kv['v'] = v.detach()

    def group_query(self, hidden_states):
        B, L, H = hidden_states.shape

        query = self.query_token.expand(B, -1, -1) #[Batch, 1, Hidden]

        attn_output, _ = self.query_attention(
            query = query,
            key = hidden_states,
            value = hidden_states
        )

        pooled = attn_output.squeeze(1)

        return pooled #[B, H]
        


    def forward(self, input_ids: torch.LongTensor, targets=None):
        """
        Runs: embed -> layers 0..pause_layer-1 -> pause & AH retrieval -> continue layers ->
        apply MSI at inject_layers -> final norm & lm_head -> logits
        """
        B, seq_len = input_ids.shape

        # embeddings (use underlying model's embedding)
        hidden_states = self.transformer.embed_tokens(input_ids).to(self.device)  # [B, L, H]
        self.embeddings = hidden_states

        B, L, H = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Create an attention mask of dimension [Batch, 1, L, L]
        # causal_mask = create_answer_only_mask(input_ids, tokenizer, device=device, dtype=dtype)
        attention_mask = create_answer_only_mask(input_ids, self.tokenizer, device, dtype)

        position_ids = torch.arange(0, L, device=device).unsqueeze(0).expand(B, L).long()  # [B, L]

        rotary_embd = self.transformer.rotary_emb
        position_embeddings = rotary_embd(hidden_states, position_ids)

        # iterate transformer layers manually
        num_layers = len(self.transformer.layers)

        mem_vec = None
              
        # run layers up to pause_layer-1 inclusive
        for i, layer in enumerate(self.transformer.layers):
            hidden_states = layer(hidden_states, 
                                 attention_mask=attention_mask,
                                 position_ids=position_ids,
                                 position_embeddings=position_embeddings
                                 )

            # hidden_states = layer_output[0]

            # if we've reached the layer before pause_layer, stop and retrieve
            if (i + 1) == self.pause_layer:
                query_pre = self.group_query(hidden_states)
                # project to AH query dim
                query = self.projection_head(query_pre)  # [B, q_dim]
                # Synchronous AH retrieval
                mem_vec = self.ah.retrieve(query, self.captured_kv)  # should be [B, H]
                # ensure mem_vec on same device and dtype
                mem_vec = mem_vec.to(device).type(dtype)
                # break out and continue from this exact hidden_states
                # note: do not apply MSI here
                break


        # Now continue remaining layers (from i+1 ... end)
        start_idx = i + 1
        for j in range(start_idx, num_layers):
            layer = self.transformer.layers[j]
            layer_outputs = layer(hidden_states, 
                                  attention_mask=attention_mask, 
                                  position_ids=position_ids,
                                  position_embeddings=position_embeddings)
            

            # apply MSI after this layer if configured
            if j in self.inject_layers:
                hidden_states = self.msi_layers[str(j)](hidden_states, mem_vec)

        # final layer norm & lm head (use base model's heads)
        hidden_states = self.transformer.norm(hidden_states)
        logits = self.base.lm_head(hidden_states)

        if targets:

            loss = F.cross_entropy(logits, targets, ignore_index=-100, reduction='mean')
        else:
            loss = 0

        return logits, loss

    # convenience method: decode via base tokenizer/generate without memory integration (use with care)
    def generate_with_memory(self, input_ids, max_new_tokens=256):
        # You need a custom generation loop to integrate AH at each step.
        # For now this delegates to base generate (no AH integration). Use only for testing.
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            print("PROBS: ",probs[:5])


            idx_next = probs.argmax(dim=-1, keepdim=True) #only take token with highest probability
            # append sampled index to the running sequence
            input_ids = torch.cat((input_ids, idx_next), dim=1) # (B, T+1)

        return input_ids
    
    def __del__(self):
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()


class RewardModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, query, response): #Receives the embeddings

        embs = torch.cat([query, response], dim=-1)

        attn_scores = self.attn(embs)
        weights = F.softmax(embs, dim=1)

        pooled = weights * embs
        pooled = F.normalize(pooled, dim=-1)

        r = self.net(pooled).squeeze(-1)

        return r
    

def jitter(memory, priority):
    sigma = 0.03 + 0.05 * priority

    memory = memory + torch.randn_like(memory) * sigma * memory.std(dim=0, keepdim=True)

    return memory


def store_memories(memories, rewards, novelties, buffer):
    reward_beta = 1e-5

    for memory, novelty, reward in zip(memories, novelties, rewards):
        priority = novelty * (1 + reward_beta * F.sigmoid(reward))
        buffer.add(memory, priority)



def consolidate(mem_buffer, sae:SAE_LTM, n_epochs):

    sae.optimizer.lr = sae.lr * 10 #boost lr for consolidation

    for epoch in range(n_epochs):

        batch, priorities = mem_buffer.get()

        batch_input = jitter(batch)
        recons, _ = sae(batch_input)

        for recon, target, priority in zip(recons, batch, priorities):
            loss = sae.loss(recon, target)

            weighted_loss = loss * priority
            weighted_loss.backward()

            sae.optimizer.step()
            sae.optimizer.zero_grad()

            

        


def stage_1_training():

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    tokenizer.padding_side = 'right'
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    args = AH_Args()
    ah = AH(args).to(device)

    model = LlamaWithAH(
        base_model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        tokenizer=tokenizer,
        ah_module=ah,
        args=args,
        inject_layers=[15,16,17,18],
        device=device
        ).to(device)
    
    # Freeze LLM weights
    for p in model.base.parameters():
        p.requires_grad = False

    # Collect trainable params (AH + MSI)
    trainable_params = list(model.ah.parameters()) + list(model.msi_layers.parameters()) + list(model.projection_head.parameters())

    optimizer = optim.AdamW(trainable_params, lr=1e-4)

    total_params = sum(p.numel() for p in trainable_params)

    trivia = load_dataset("trivia_qa", "rc.nocontext")
    trivia = load_dataset("trivia_qa", "rc.nocontext")  # "rc" = reading comprehension version


    # Access splits
    train_data = trivia["train"]

    train_data = train_data.remove_columns(['question_id', 'question_source', 'entity_pages', 'search_results'])
    questions = train_data['question']
    answers = train_data['answer']
    answers = [answer['value'] for answer in answers]

    separator = ' Answer:'
    dset = Stage1_Dataset(questions, answers, separator)
    dataloader = DataLoader(dset, batch_size=args.batch_size)

    mem_buffer = PrioritizedMemory(batch_size=args.batch_size)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
    consolidate_count = 0
    for batch in dataloader:
        
        # inputs = preprocess_fn(batch, tokenizer, max_len=256)
        inputs = tokenizer(batch, truncation=True, padding=True, return_tensors='pt', max_length=128)
        input_ids = inputs.input_ids.to(device)
        batch_size, seq_len = input_ids.shape
        targets = input_ids.clone()

        separator_ids = tokenizer.encode(separator, add_special_tokens=False)
    
        for b in range(args.batch_size):
            # Find where answer starts (after separator)
            answer_start = None
            for i in range(seq_len - len(separator_ids)):
                if input_ids[b, i:i+len(separator_ids)].tolist() == separator_ids:
                    answer_start = i + len(separator_ids)
                    break
            
            if answer_start is not None:
                # Mask prompt tokens (set to -100 so they're ignored by CrossEntropyLoss)
                targets[b, :answer_start] = -100
            else:
                # If separator not found, mask everything (safety)
                targets[b, :] = -100
            
        # Shift for next-token prediction
        targets = targets[:, 1:].contiguous()
        answer_mask = (targets != -100)
        # attn_mask = inputs.attention_mask.to(device)
        
        
        tokens, loss = model.forward(input_ids, targets) #GET TARGETS
        input_embds = model.embeddings
        memories = model.ah.ltm.targets
        novelties = model.ah.stage_1.novelty

        #Decode
        """
        We first need our token values of ONLY the completion and then get the embeddings of the completion.
        We pass the embeddings of our prompt and completion into our memory trace and reward model
        then we decode and pass to user (Or we could pass to user first and while they read we do the embeddings)
        """
        completions = tokenizer.batch_decode(tokens) 


        #Get Feedback 

        feedback = None #We need to get user response for the memory trace. Maybe make feedback optional
        """

        
        Thoughts on Feedback:

        When creating a memory, we need to know if what we said went over well to know if our memory is working
        well or not. An essential part of a memory is the result of our output. Having a feedback signal will
        help us get a gauge of where we are in terms of memory, reasoning, and the integration of these two.
        It will also let us know how "important" a memory is alongside its novelty as a memory that results
        in an extreme feedback signal either way should be remembered well.

        Currently, a reward model seems the most straight-forward method for getting a reward signal. An 
        important detail is that we will need to figure out how to interpret no direct feedback. 
        No direct feedback on a question is usually a good thing sincce this means that we answered the 
        question in its entirety and there is no need for further dialogue, but there may be instances
        where no feedback is a bad thing, so we will need to train our model on these scenarios.
        """
        reward_m = RewardModel(model.args.embd_dim, hidden_dim=512)
        rewards = reward_m(input_embds, completions)
        #Create memory trace
        mem_trace = MemoryTrace(input_ids, completions, model.args.embd_dim, model.args.storage_dim, feedback, rewards)
        mem_trace = mem_trace.create_trace(model)
        
        store_memories(memories, rewards, novelties, mem_buffer)
        consolidate_count += 1

        if consolidate_count >= 7:
            consolidate(mem_buffer, model.ah.ltm, n_epochs = 5)
            consolidate_count = 0
        
        model.ah.stage_1.update(model.ah.ent_out, mem_trace, novelties, rewards)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"loss: {loss.item():.4f}")


if __name__ == "__main__":
    stage_1_training()

"""
PROMPTS FOR TOMORROW:

1) For my reward model, my desire is for it to receive the prompt and completion embedding
and then output a scalar for the predicted reward. Thus, it would be trained with inputs of prompt and completion embeddings and a scalar target. However, it seems to be common practice 
for the reward model to learn to prefer certain types of responses and to output based on preference. Should I stick with my current plan to train a reward model to output a simple scalar or should I adopt a prefernce-style reward model?

2) Here is my current training plan:
Stage 1
I will first need to get a working reward model (hopefully we are able to find a pretrained one that outputs scalar rewards as desired). Once I have this, I will be freezeing the parameters of the language model and only training my AH parameters. I will be using the TriviaQA dataset for my stage 1 training and loss will be calculated on answers only. This will hopefully teach the AH to store facts and reintegrate them will the LLM.

Stage 2
In stage 2 I will unfreeze the LLM weights and begin training the whole system. I will be using the KILT dataset for stage 2. This should hopefully train the system to work as a whole and the AH will maintain its memory-storage role as it was taught in stage 1, and the LLM will learn to better utilize in Stage 2

Stage 3
Stage 3 will be RLHF, primarily for alignment and small semantic changes.

I'm concerned with my Stage 1 as I'm not sure this training method will work well with my model set up. 
The goal of my training is for the model to learn to work as a whole, letting the AH be responsible for memory storage and the LLM responsible for reasoning. 

Deeply analyze this train plan and provide strengths and weaknesses. If there are significant weaknesses or inefficiencies, provide a more optimal training plan.
"""