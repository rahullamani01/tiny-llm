import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """Step 1 baseline. Will be replaced by the full TinyGPT in later steps."""

    def __init__(self, vocab_size):
        super().__init__()
        # row i = the scores for "which char comes after char i"
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding(idx)  # (B, T, vocab)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]                          # only the last position matters
            probs = F.softmax(logits, dim=-1)                  # scores -> probabilities
            next_id = torch.multinomial(probs, num_samples=1)  # sample one token
            idx = torch.cat([idx, next_id], dim=1)             # append, feed back in
        return idx
