"""Sampling utilities: turn next-token logits into a token id.

The tricks here are the usual knobs on top of the raw softmax:

* temperature        -> flatten (wilder) or sharpen (safer) the distribution
* top-k              -> keep only the k most likely tokens
* top-p (nucleus)    -> keep the smallest set of tokens whose probability sums to p
* repetition penalty -> push down tokens that already appeared in the context
"""
import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


def _validate(temperature, top_k, top_p, repetition_penalty):
    if temperature < 0:
        raise ValueError("temperature must be >= 0 (0 means greedy)")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1, or None to disable it")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1], or None to disable it")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be > 0")


def apply_repetition_penalty(logits, generated, penalty):
    """Divide positive logits of seen tokens by `penalty`, multiply negative ones.

    This is the CTRL-style penalty: a penalty > 1 makes already generated tokens
    less likely, while leaving unseen tokens untouched.
    """
    if penalty == 1.0 or generated is None or generated.numel() == 0:
        return logits

    logits = logits.clone()
    for row in range(logits.size(0)):
        seen = torch.unique(generated[row])
        scores = logits[row, seen]
        scores = torch.where(scores > 0, scores / penalty, scores * penalty)
        logits[row, seen] = scores
    return logits


def apply_top_k(logits, k):
    """Mask every logit below the k-th largest one."""
    if k is None:
        return logits
    k = min(k, logits.size(-1))
    cutoff = torch.topk(logits, k, dim=-1).values[:, [-1]]
    return logits.masked_fill(logits < cutoff, NEG_INF)


def apply_top_p(logits, p):
    """Keep the smallest set of tokens whose cumulative probability reaches p."""
    if p is None or p >= 1.0:
        return logits

    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)

    # `cumulative - probs` is the mass *before* the current token. Drop tokens
    # once that mass already exceeds p, but always keep the most likely one.
    drop = cumulative - probs > p
    drop[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(drop, NEG_INF)
    return sorted_logits.scatter(1, sorted_idx, sorted_logits)


@torch.no_grad()
def sample_next_token(
    logits,
    temperature=1.0,
    top_k=None,
    top_p=None,
    repetition_penalty=1.0,
    generated=None,
):
    """Sample one token per batch row from logits of shape (B, vocab).

    Args:
        logits: unnormalised next-token scores, shape (B, vocab).
        temperature: 0 selects the argmax (greedy); higher is more random.
        top_k: keep only the k most likely tokens (None/0 disables).
        top_p: nucleus cutoff in (0, 1] (None/1.0 disables).
        repetition_penalty: > 1 discourages tokens in `generated`.
        generated: token ids seen so far, shape (B, T), used by the penalty.

    Returns:
        Long tensor of shape (B, 1) with the sampled token ids.
    """
    _validate(temperature, top_k, top_p, repetition_penalty)

    logits = apply_repetition_penalty(logits, generated, repetition_penalty)
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)
