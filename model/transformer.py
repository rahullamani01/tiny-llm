import torch.nn as nn
from model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Communication (attention) followed by computation (feed-forward)."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.0,
                 pos_embedding="learned", rope_base=10_000.0):
        super().__init__()
        self.sa = MultiHeadAttention(
            n_embd,
            n_head,
            block_size,
            dropout,
            pos_embedding=pos_embedding,
            rope_base=rope_base,
        )
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # residual connection around attention
        x = x + self.ffwd(self.ln2(x))   # residual connection around feed-forward
        return x

    def forward_with_cache(self, x, past_kv=None, pos_offset=0):
        """Forward one block and return its attention cache for this layer."""
        attn, present = self.sa.forward_with_cache(self.ln1(x), past_kv, pos_offset)
        x = x + attn
        x = x + self.ffwd(self.ln2(x))
        return x, present
