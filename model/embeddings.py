import torch
import torch.nn as nn


class Embeddings(nn.Module):
    """Token embedding plus learned positions (or no positions for RoPE)."""

    def __init__(self, vocab_size, block_size, n_embd, pos_embedding="learned"):
        super().__init__()
        if pos_embedding not in ("learned", "rope"):
            raise ValueError(f"unknown pos_embedding {pos_embedding!r}")
        self.block_size = block_size
        self.pos_embedding = pos_embedding
        self.token_embedding = nn.Embedding(vocab_size, n_embd)     # what the token is
        # RoPE injects positions inside attention instead, so it has no table.
        self.position_embedding = (
            nn.Embedding(block_size, n_embd) if pos_embedding == "learned" else None
        )

    def forward(self, idx, pos_offset=0):
        """Embed tokens at positions pos_offset .. pos_offset + T - 1.

        Decoding with a KV cache only passes the new token(s), so `pos_offset`
        keeps positions in sync with the full sequence. Learned mode adds the
        position vector here; RoPE mode adds nothing (attention rotates q/k).
        """
        B, T = idx.shape
        if pos_offset < 0:
            raise ValueError("pos_offset must be >= 0")
        if pos_offset + T > self.block_size:
            raise ValueError(
                f"positions {pos_offset}..{pos_offset + T - 1} exceed "
                f"block_size={self.block_size}"
            )
        tok = self.token_embedding(idx)                                     # (B, T, C)
        if self.position_embedding is None:  # RoPE
            return tok
        pos = self.position_embedding(
            torch.arange(pos_offset, pos_offset + T, device=idx.device)
        )                                                                   # (T, C)
        return tok + pos
