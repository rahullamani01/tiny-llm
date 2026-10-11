# Later: load a checkpoint and generate text (temperature, top-k)
import argparse

import torch

from tokenizer.tokenizer import CharTokenizer
from model.model import TinyGPT


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(path="checkpoints/tinygpt.pt", device=None):
    """Load a trained TinyGPT and its tokenizer from a checkpoint."""
    device = device or get_device()
    ckpt = torch.load(path, map_location=device)
    model = TinyGPT(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CharTokenizer.from_chars(ckpt["chars"])
    return model, tok


def generate_text(model, tok, prompt, max_new_tokens=500, temperature=0.8, top_k=40):
    device = next(model.parameters()).device
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return tok.decode(out[0].tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/tinygpt.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    a = p.parse_args()

    model, tok = load_model(a.ckpt)
    try:
        print(generate_text(model, tok, a.prompt, a.tokens, a.temperature, a.top_k))
    except KeyError as e:
        print(f"Prompt contains a character the model never saw in training: {e}")