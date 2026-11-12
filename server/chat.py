"""Chat request validation and streaming event generation.

Kept separate from the FastAPI app so the core streaming logic can be tested
without web dependencies.
"""
import torch

from chat.template import build_chat_prompt
from utils import set_seed


def validate_sampling(
    max_new_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=None,
    repetition_penalty=1.1,
):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1, or null to disable it")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1], or null to disable it")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be > 0")


def iter_chat_events(
    model,
    tok,
    messages,
    system=None,
    max_new_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=None,
    repetition_penalty=1.1,
    use_cache=True,
    seed=None,
):
    """Yield ``{"token": ...}`` events and one final ``{"done": True, ...}``.

    The final event also carries the complete reply, which is convenient for
    clients that missed a token or simply want the assembled text.
    """
    validate_sampling(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    ids = build_chat_prompt(
        messages, tok, model.block_size, add_generation_prompt=True, system=system
    )
    device = next(model.parameters()).device
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    if seed is not None:
        set_seed(seed)
    model.eval()

    pieces = []
    with torch.no_grad():
        for out in model.generate_stream(
            idx,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_cache=use_cache,
        ):
            token_id = out[0, -1].item()
            if tok.is_special_id(token_id):
                break
            piece = tok.decode([token_id])
            pieces.append(piece)
            yield {"token": piece}

    yield {"done": True, "content": "".join(pieces)}
