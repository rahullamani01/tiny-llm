"""Load a checkpoint and generate text (temperature, top-k/top-p, streaming)."""
import argparse

import torch

from checkpoint import load_checkpoint
from tokenizer import CharTokenizer, tokenizer_from_state
from model.model import TinyGPT
from utils import get_device, set_seed


def load_model(path="checkpoints/tinygpt.pt", device=None):
    """Load a trained TinyGPT and its tokenizer from a checkpoint."""
    device = device or get_device()
    ckpt = load_checkpoint(path, device)
    model = TinyGPT(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    state = ckpt.get("tokenizer")
    if state is not None:
        tok = tokenizer_from_state(state)
    else:  # legacy char-tokenizer checkpoint
        tok = CharTokenizer.from_chars(ckpt["chars"])
    return model, tok


def _encode_prompt(tok, prompt, device):
    if not prompt:
        raise ValueError("prompt must not be empty")
    missing = tok.missing_chars(prompt)
    if missing:
        chars = ", ".join(repr(c) for c in missing)
        raise ValueError(f"prompt contains characters the model never saw: {chars}")
    return torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)


def generate_text(
    model,
    tok,
    prompt,
    max_new_tokens=500,
    temperature=0.8,
    top_k=40,
    top_p=None,
    repetition_penalty=1.0,
    seed=None,
):
    """Generate a completion and return the prompt plus the new text."""
    device = next(model.parameters()).device
    idx = _encode_prompt(tok, prompt, device)
    if seed is not None:
        set_seed(seed)
    out = model.generate(
        idx,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return tok.decode(out[0].tolist())


def stream_text(
    model,
    tok,
    prompt,
    max_new_tokens=500,
    temperature=0.8,
    top_k=40,
    top_p=None,
    repetition_penalty=1.0,
    seed=None,
):
    """Yield one decoded token at a time as the model samples it."""
    device = next(model.parameters()).device
    idx = _encode_prompt(tok, prompt, device)
    if seed is not None:
        set_seed(seed)
    for out in model.generate_stream(
        idx,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    ):
        yield tok.decode(out[0, -1:].tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate text from a TinyGPT checkpoint.")
    p.add_argument("--ckpt", default="checkpoints/tinygpt.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=40, help="0 disables top-k")
    p.add_argument("--top-p", type=float, default=None, help="nucleus cutoff (off by default)")
    p.add_argument("--repetition-penalty", type=float, default=1.0, help="1.0 = off")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stream", action="store_true", help="print tokens as they are sampled")
    a = p.parse_args()

    model, tok = load_model(a.ckpt)
    opts = dict(
        max_new_tokens=a.tokens,
        temperature=a.temperature,
        top_k=a.top_k or None,
        top_p=a.top_p,
        repetition_penalty=a.repetition_penalty,
        seed=a.seed,
    )
    try:
        if a.stream:
            for chunk in stream_text(model, tok, a.prompt, **opts):
                print(chunk, end="", flush=True)
            print()
        else:
            print(generate_text(model, tok, a.prompt, **opts))
    except ValueError as e:
        raise SystemExit(f"error: {e}")
