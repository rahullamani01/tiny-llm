import torch
import torch.nn as nn


class Embeddings(nn.Module):
    """Token embedding + learned position embedding."""

    def __init__(self, vocab_size, block_size, n_embd):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)     # what the token is
        self.position_embedding = nn.Embedding(block_size, n_embd)  # where it sits

    def forward(self, idx, pos_offset=0):
        """Embed tokens at absolute positions pos_offset .. pos_offset + T - 1.

        Decoding with a KV cache only passes the new token(s), so `pos_offset`
        keeps the learned position embedding in sync with the full sequence.
        """
        B, T = idx.shape
        limit = self.position_embedding.num_embeddings
        if pos_offset < 0:
            raise ValueError("pos_offset must be >= 0")
        if pos_offset + T > limit:
            raise ValueError(
                f"positions {pos_offset}..{pos_offset + T - 1} exceed block_size={limit}"
            )
        tok = self.token_embedding(idx)                                     # (B, T, C)
        pos = self.position_embedding(
            torch.arange(pos_offset, pos_offset + T, device=idx.device)
        )                                                                   # (T, C)
        return tok + pos
