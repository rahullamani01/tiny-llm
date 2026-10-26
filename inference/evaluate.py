"""Evaluate a checkpoint: cross-entropy loss and perplexity."""
import argparse
import math

import torch

from inference.generate import load_model
from training.dataset import split_data
from training.evaluate import evaluate_loss
from utils import get_device, set_seed


def main():
    p = argparse.ArgumentParser(description="Evaluate a TinyGPT checkpoint.")
    p.add_argument("--ckpt", default="checkpoints/tinygpt.pt")
    p.add_argument("--data", default="data/training.txt")
    p.add_argument("--split", choices=["train", "val", "all"], default="val")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    a = p.parse_args()

    try:
        device = get_device(a.device)
        model, tok = load_model(a.ckpt, device)
    except ValueError as e:
        raise SystemExit(f"error: {e}")

    try:
        text = open(a.data, encoding="utf-8").read()
    except FileNotFoundError:
        raise SystemExit(f"error: data file not found: {a.data}")

    missing = tok.missing_chars(text)
    if missing:
        shown = ", ".join(repr(c) for c in missing)
        raise SystemExit(f"error: data contains characters the model never saw: {shown}")

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    if a.split == "all":
        chosen = data
    else:
        try:
            chosen = split_data(data, a.val_fraction)[a.split]
        except ValueError as e:
            raise SystemExit(f"error: {e}")

    set_seed(a.seed)
    try:
        loss = evaluate_loss(
            model,
            chosen,
            model.block_size,
            batch_size=a.batch_size,
            iters=a.iters,
            device=device,
        )
    except ValueError as e:
        raise SystemExit(f"error: {e}")

    print(f"checkpoint: {a.ckpt}")
    print(f"split: {a.split} | tokens: {len(chosen)} | device: {device}")
    print(f"loss: {loss:.4f}")
    print(f"perplexity: {math.exp(loss):.2f}")


if __name__ == "__main__":
    main()
