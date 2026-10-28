import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import Head
from model.embeddings import Embeddings
from model.sampling import sample_next_token
from model.transformer import Block


def _loss(logits, targets):
    B, T, C = logits.shape
    return F.cross_entropy(logits.view(B * T, C), targets.view(B * T))


class BigramLanguageModel(nn.Module):
    """Step 1 baseline: predicts the next char from the current char only."""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding(idx)
        loss = None if targets is None else _loss(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


class AttentionLanguageModel(nn.Module):
    """Step 3: embeddings + a single self-attention head."""

    def __init__(self, vocab_size, block_size, n_embd):
        super().__init__()
        self.block_size = block_size
        self.embed = Embeddings(vocab_size, block_size, n_embd)
        self.sa_head = Head(n_embd, n_embd, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        x = self.sa_head(x)
        logits = self.lm_head(x)
        loss = None if targets is None else _loss(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


class TinyGPT(nn.Module):
    """Embeddings -> N transformer blocks -> LayerNorm -> output head."""

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout=0.0,
                 tie_weights=False, pos_embedding="learned", rope_base=10_000.0):
        super().__init__()
        if pos_embedding not in ("learned", "rope"):
            raise ValueError(f"unknown pos_embedding {pos_embedding!r}")
        self.block_size = block_size
        self.pos_embedding = pos_embedding
        self.embed = Embeddings(
            vocab_size, block_size, n_embd, pos_embedding=pos_embedding
        )
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    n_head,
                    block_size,
                    dropout,
                    pos_embedding=pos_embedding,
                    rope_base=rope_base,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
        if tie_weights:
            # Share the token embedding and output projection (GPT-2 style).
            self.lm_head.weight = self.embed.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_key_values=None, use_cache=False):
        """Run the transformer.

        Returns ``(logits, loss)`` by default. With ``use_cache=True`` it
        returns ``(logits, loss, present)``, where ``present`` is a tuple with
        one ``(keys, values)`` pair per layer, each shaped
        ``(B, n_head, cached_tokens, head_size)``.

        Pass a previous ``present`` as ``past_key_values`` to run only the new
        tokens. Positions continue after the cached tokens, so the cache length
        must stay within ``block_size``.
        """
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if past_key_values is None:
            pos_offset = 0
        else:
            if len(past_key_values) != len(self.blocks):
                raise ValueError(
                    f"expected {len(self.blocks)} layer caches, "
                    f"got {len(past_key_values)}"
                )
            pos_offset = past_key_values[0][0].shape[2]

        x = self.embed(idx, pos_offset=pos_offset)  # (B, T, n_embd)
        presents = []
        for layer, block in enumerate(self.blocks):
            if use_cache:
                past = None if past_key_values is None else past_key_values[layer]
                x, present = block.forward_with_cache(x, past, pos_offset)
                presents.append(present)
            else:
                x = block(x)                        # N x (attention + feed-forward)
        x = self.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab)
        loss = None if targets is None else _loss(logits, targets)
        if use_cache:
            return logits, loss, tuple(presents)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        use_cache=False,
    ):
        """Sample tokens one at a time and return the full sequence.

        Call model.eval() first. See model.sampling for the meaning of the
        sampling options. Set ``use_cache=True`` to reuse keys/values between
        decoding steps; the output matches the uncached path.
        """
        out = idx
        for out in self.generate_stream(
            idx,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_cache=use_cache,
        ):
            pass
        return out

    def generate_stream(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        use_cache=False,
    ):
        """Yield the full sequence after each newly sampled token.

        Useful for streaming output: read the last token of each yield.

        With a KV cache, the prompt is prefilled once and later steps forward
        only the newest token. The learned absolute position embeddings force a
        cache rebuild whenever the context window has to slide: the retained
        tokens are re-run from position 0, exactly like the uncached generator.
        """
        with torch.no_grad():
            past_key_values = None
            for _ in range(max_new_tokens):
                if use_cache:
                    if past_key_values is None or idx.size(1) > self.block_size:
                        # First step, or the window moved: positions restart at 0,
                        # so recompute the retained context instead of reusing it.
                        window = idx[:, -self.block_size:]
                        logits, _, past_key_values = self(window, use_cache=True)
                    else:
                        logits, _, past_key_values = self(
                            idx[:, -1:],
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                else:
                    idx_cond = idx[:, -self.block_size:]  # crop to the context window
                    logits, _ = self(idx_cond)

                next_id = sample_next_token(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generated=idx,
                )
                idx = torch.cat([idx, next_id], dim=1)
                yield idx
