# Step 2: token embeddings + position embeddings
import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)     # what the token is
        self.position_embedding = nn.Embedding(block_size, n_embd)  # where it sits

    def forward(self, idx):
        B, T = idx.shape
        tok = self.token_embedding(idx)                                     # (B, T, C)
        pos = self.position_embedding(torch.arange(T, device=idx.device))  # (T, C)
        return tok + pos
    