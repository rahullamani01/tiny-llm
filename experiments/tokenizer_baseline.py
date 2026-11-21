"""Small, repeatable character vs BPE training experiment."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from inference.generate import generate_text
from model.model import TinyGPT
from tokenizer import BPETokenizer, CharTokenizer
from training.dataset import get_batch
from training.evaluate import evaluate_loss
from utils import get_device, set_seed


def split_text(text, val_fraction):
    """Split before tokenization, so both models see the same raw text."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    n = int(len(text) * (1 - val_fraction))
    if n <= 0 or n >= len(text):
        raise ValueError("text is too short for the requested split")
    return text[:n], text[n:]


def run_one(kind, train_text, val_text, args, device):
    start = time.perf_counter()
    if kind == "char":
        tok = CharTokenizer(train_text)
    else:
        tok = BPETokenizer.train(train_text, args.vocab_size, max_chars=args.bpe_max_chars)
    missing = tok.missing_chars(val_text)
    if missing:
        raise ValueError(f"validation contains characters absent from training: {missing!r}")
    train = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val = torch.tensor(tok.encode(val_text), dtype=torch.long)
    if min(len(train), len(val)) <= args.block_size:
        raise ValueError("each split must contain more tokens than block_size")
    preparation_seconds = time.perf_counter() - start

    set_seed(args.seed)
    model = TinyGPT(tok.vocab_size, args.block_size, args.n_embd, args.n_head,
                    args.n_layer, dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = time.perf_counter()
    model.train()
    for _ in range(args.steps):
        x, y = get_batch(train, args.block_size, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start

    train_loss = evaluate_loss(model, train, args.block_size, args.batch_size,
                               args.eval_iters, device, seed=args.seed + 1)
    val_loss = evaluate_loss(model, val, args.block_size, args.batch_size,
                             args.eval_iters, device, seed=args.seed + 2)
    model.eval()
    samples = {}
    for prompt in args.prompts:
        samples[prompt] = generate_text(model, tok, prompt, args.sample_tokens,
                                        temperature=0, top_k=None)
    return {
        "tokenizer": kind,
        "vocab_size": tok.vocab_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_tokens": len(train),
        "val_tokens": len(val),
        "train_chars_per_token": len(train_text) / len(train),
        "val_chars_per_token": len(val_text) / len(val),
        "preparation_seconds": preparation_seconds,
        "training_seconds": training_seconds,
        "train_loss_nats_per_token": train_loss,
        "val_loss_nats_per_token": val_loss,
        "samples_greedy": samples,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/training.txt")
    p.add_argument("--output", required=True, help="new JSON file; never overwrites")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--n-embd", type=int, default=64)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--bpe-max-chars", type=int, default=20000)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--sample-tokens", type=int, default=40)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.block_size, args.n_embd, args.n_head,
           args.n_layer, args.vocab_size, args.bpe_max_chars, args.eval_iters) < 1:
        p.error("sizes, steps and evaluation iterations must be positive")
    if args.n_embd % args.n_head or args.lr <= 0 or args.sample_tokens < 0:
        p.error("embedding size must divide by heads; lr must be positive and sample tokens nonnegative")
    args.prompts = ["ROMEO:\n", "JULIET:\n"]
    path = Path(args.output)
    if path.exists():
        p.error(f"output already exists: {path}")
    raw = Path(args.data).read_bytes()
    train_text, val_text = split_text(raw.decode("utf-8"), args.val_fraction)
    device = get_device(args.device)
    result = {
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_bytes": len(raw),
        "train_characters": len(train_text),
        "val_characters": len(val_text),
        "config": vars(args),
        "device": str(device),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "results": [],
    }
    for kind in ("char", "bpe"):
        print(f"running {kind}...", flush=True)
        metrics = run_one(kind, train_text, val_text, args, device)
        result["results"].append(metrics)
        print(f"  val loss/token {metrics['val_loss_nats_per_token']:.4f}, "
              f"training {metrics['training_seconds']:.1f}s", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
