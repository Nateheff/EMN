import torch
import torch.nn as nn

"""
Notes on Episode creation:

We first need to implement a method of reading/tracking a conversation
Then, at each generation we need to save the log prob distribution to us for the current
kl-divergence (Bayesian surprise) and the next one to do:
    surprise = KL(P(*|context[0:t]) | P(*|context[0:t-1]))
then once suprise surpasses our threshold (which we need to implement), we set the episode boundary and save

TO DO:
 - Implement conversation storage/reading
 - Implement saving log prob for current and previous distribution
 - implement surprise calcualtion
 - implement surprise threshold

THEN  we can come here and implement our actual episode representation:
We want our episodes in working memory to be compressed yet reconstuctible for 
consolidation and quickly accessible by content and relevance. 

First, we'll handle creatig the compressed yet reconstructible episode representations:
"""

def encode_long_episode(self, episode_text: str, metadata: dict) -> torch.Tensor:
    """
    Chunk → Embed → Aggregate for episodes exceeding context length
    """
    # Split into chunks at natural boundaries (turns, not mid-sentence)
    chunks = self.intelligent_chunking(episode_text, max_tokens=400)
    
    # Encode each chunk
    chunk_embeddings = self.encoder.encode(
        chunks,
        batch_size=len(chunks),
        convert_to_tensor=True,
        show_progress_bar=False
    )  # [num_chunks, 768]
    
    # Aggregate with surprise-weighted attention
    chunk_surprises = self.extract_chunk_surprises(metadata)
    episode_embedding = self.aggregator(chunk_embeddings, chunk_surprises)
    
    return episode_embedding

def intelligent_chunking(self, text: str, max_tokens: int):
    """
    Chunk at turn boundaries, not arbitrary positions
    Maintains conversational coherence
    """
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for line in lines:
        line_length = len(self.encoder.tokenize([line])[0])
        
        if current_length + line_length > max_tokens and current_chunk:
            # Start new chunk at turn boundary
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_length = line_length
        else:
            current_chunk.append(line)
            current_length += line_length
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

class AttentionPooling(nn.Module):
    """
    Lightweight attention aggregation guided by surprise scores
    Based on pooling research: mixed strategies outperform single methods
    """
    def __init__(self, d_model=768):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))  # Learnable "importance" query
        self.projection = nn.Linear(d_model, d_model)
        
    def forward(self, chunk_embeds, chunk_surprises=None):
        # chunk_embeds: [num_chunks, d_model]
        # chunk_surprises: [num_chunks] or None
        
        # Compute attention scores
        attention_logits = torch.matmul(chunk_embeds, self.query)  # [num_chunks]
        
        # Modulate by surprise (high surprise → higher weight)
        if chunk_surprises is not None:
            surprise_weights = torch.softmax(chunk_surprises, dim=0)
            attention_logits = attention_logits + torch.log(surprise_weights + 1e-8)
        
        attention_weights = torch.softmax(attention_logits, dim=0)  # [num_chunks]
        
        # Weighted aggregation
        episode_embed = torch.sum(
            chunk_embeds * attention_weights.unsqueeze(-1),
            dim=0
        )  # [d_model]
        
        # Final projection
        episode_embed = self.projection(episode_embed)
        
        return episode_embed

"""
Once we have our compressed/embedded representation, we need to store it somehow.I feel like I'm going about this all wrong.
"""