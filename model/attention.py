import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)    # "what do I contain?"
        self.query = nn.Linear(n_embd, head_size, bias=False)  # "what am I looking for?"
        self.value = nn.Linear(n_embd, head_size, bias=False)  # "what do I pass on?"
        # lower-triangular matrix: position t may only look at positions <= t
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.last_attn = None  # saved for attention_visualization.py

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                               # (B, T, hs)
        q = self.query(x)                                             # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5           # (B, T, T) scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # hide the future
        wei = F.softmax(wei, dim=-1)                                  # scores -> weights
        self.last_attn = wei.detach()
        wei = self.dropout(wei)
        v = self.value(x)                                             # (B, T, hs)
        return wei @ v                                                # weighted sum of values


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, each in its own smaller subspace."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)  # mix the heads' outputs together
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        return self.dropout(self.proj(out))