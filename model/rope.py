"""Rotary position embeddings (RoPE).

Instead of adding a learned position vector to the token embedding, RoPE
rotates each query and key vector by an angle proportional to its position.
The dot product of two rotated vectors then depends only on their relative
distance, which is why RoPE works naturally with incremental decoding.

Each head's vector is split into adjacent pairs ``(x[2i], x[2i+1])`` and each
pair is rotated by ``pos * inv_freq[i]``. GPT-NeoX uses this interleaved
layout; other implementations sometimes split the vector in half instead.
"""
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, block_size, base=10_000.0):
        super().__init__()
        if head_size % 2 != 0:
            raise ValueError("RoPE requires an even head_size")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")

        self.head_size = head_size
        self.block_size = block_size

        # inv_freq[i] = 1 / base^(2i / head_size), one frequency per pair.
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for every position we can ever use.
        positions = torch.arange(block_size, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)          # (block_size, head_size/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x, pos_offset=0):
        """Rotate the last dimension of `x` by positions pos_offset..+T-1.

        Args:
            x: tensor shaped (..., T, head_size).
            pos_offset: absolute position of the first token in `x`.

        Returns:
            A tensor with the same shape as `x`.
        """
        T = x.size(-2)
        if pos_offset < 0 or pos_offset + T > self.block_size:
            raise ValueError(
                f"positions {pos_offset}..{pos_offset + T - 1} exceed "
                f"block_size={self.block_size}"
            )

        cos = self.cos_cached[pos_offset:pos_offset + T].to(x.dtype)
        sin = self.sin_cached[pos_offset:pos_offset + T].to(x.dtype)

        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated = torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        )
        return rotated.flatten(-2)
