import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import Head
from model.embeddings import Embeddings
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

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.embed = Embeddings(vocab_size, block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.embed(idx)        # (B, T, n_embd)
        x = self.blocks(x)         # N x (attention + feed-forward)
        x = self.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab)
        loss = None if targets is None else _loss(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Sample tokens one at a time. Call model.eval() first."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]                # crop to the context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # <1 = safer, >1 = wilder
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")     # drop all but the top k
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx