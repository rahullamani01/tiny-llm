import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import RotaryEmbedding


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, n_embd, head_size, block_size, dropout=0.0,
                 pos_embedding="learned", rope_base=10_000.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)    # "what do I contain?"
        self.query = nn.Linear(n_embd, head_size, bias=False)  # "what am I looking for?"
        self.value = nn.Linear(n_embd, head_size, bias=False)  # "what do I pass on?"
        # lower-triangular matrix: position t may only look at positions <= t
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.rope = (
            RotaryEmbedding(head_size, block_size, rope_base)
            if pos_embedding == "rope"
            else None
        )
        self.last_attn = None  # saved for attention_visualization.py

    def forward(self, x):
        return self._attend(x)

    def forward_with_cache(self, x, past_kv=None, pos_offset=0):
        """Attend to `x` plus optional cached keys/values.

        Args:
            x: (B, T_new, n_embd) hidden states for the new tokens.
            past_kv: optional (keys, values), each (B, T_cached, head_size).
            pos_offset: absolute position of the first new token. It must equal
                the number of cached tokens, because positions are absolute.

        Returns:
            (out, (keys, values)) where the returned keys/values include the
            cached tokens followed by the new ones.
        """
        return self._attend(x, past_kv=past_kv, pos_offset=pos_offset, use_cache=True)

    def _attend(self, x, past_kv=None, pos_offset=0, use_cache=False):
        B, T, C = x.shape
        q = self.query(x)                                             # (B, T, hs)
        k = self.key(x)                                               # (B, T, hs)
        v = self.value(x)                                             # (B, T, hs)

        if self.rope is not None:
            # Rotate q and k at their absolute positions. Cached keys were
            # already rotated when they were first computed.
            q = self.rope(q, pos_offset)
            k = self.rope(k, pos_offset)

        if past_kv is not None:
            past_k, past_v = past_kv
            if past_k.dim() != 3 or past_v.dim() != 3:
                raise ValueError(
                    "cached keys/values must be 3-D (batch, tokens, head_size)"
                )
            if past_k.shape != past_v.shape:
                raise ValueError("cached key and value shapes must match")
            if past_k.shape[0] != B:
                raise ValueError(
                    f"cache batch size {past_k.shape[0]} does not match input batch size {B}"
                )
            if past_k.shape[1] != pos_offset:
                raise ValueError(
                    f"pos_offset={pos_offset} must equal the cached length "
                    f"{past_k.shape[1]}"
                )
            k = torch.cat([past_k, k], dim=1)                         # (B, T_cached+T, hs)
            v = torch.cat([past_v, v], dim=1)

        T_k = k.shape[1]
        if pos_offset + T > self.tril.size(0) or T_k > self.tril.size(1):
            raise ValueError(
                f"positions {pos_offset}..{pos_offset + T - 1} with {T_k} keys "
                f"exceed block_size={self.tril.size(0)}"
            )

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5           # (B, T, T_k)
        # Query i sits at absolute position pos_offset + i, so row
        # pos_offset + i of the causal mask is the right one. This also handles
        # the one-token decode case, where a plain tril[:1, :T_k] would be wrong.
        mask = self.tril[pos_offset:pos_offset + T, :T_k]
        wei = wei.masked_fill(mask == 0, float("-inf"))               # hide the future
        wei = F.softmax(wei, dim=-1)                                  # scores -> weights
        self.last_attn = wei.detach()
        wei = self.dropout(wei)
        out = wei @ v                                                 # weighted sum of values
        if use_cache:
            return out, (k, v)
        return out


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, each in its own smaller subspace."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.0,
                 pos_embedding="learned", rope_base=10_000.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [
                Head(
                    n_embd,
                    head_size,
                    block_size,
                    dropout,
                    pos_embedding=pos_embedding,
                    rope_base=rope_base,
                )
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd)  # mix the heads' outputs together
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        return self.dropout(self.proj(out))

    def forward_with_cache(self, x, past_kv=None, pos_offset=0):
        """Like forward, but return per-layer cached keys/values.

        The returned cache is (keys, values) shaped (B, n_head, T_cached, hs).
        """
        outs, keys, values = [], [], []
        for i, head in enumerate(self.heads):
            head_past = None if past_kv is None else (past_kv[0][:, i], past_kv[1][:, i])
            out, (k, v) = head.forward_with_cache(x, head_past, pos_offset)
            outs.append(out)
            keys.append(k)
            values.append(v)

        out = torch.cat(outs, dim=-1)  # (B, T_new, n_embd)
        out = self.dropout(self.proj(out))
        return out, (torch.stack(keys, dim=1), torch.stack(values, dim=1))
