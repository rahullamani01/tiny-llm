"""Small, repeatable character vs BPE training experiment."""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from inference.generate import generate_text
from model.model import TinyGPT
from tokenizer import BPETokenizer, CharTokenizer
from training.dataset import get_batch
from utils import get_device, set_seed


def split_text(text, val_fraction):
    """Split before tokenization, so both models see the same raw text."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    n = int(len(text) * (1 - val_fraction))
    if n <= 0 or n >= len(text):
        raise ValueError("text is too short for the requested split")
    return text[:n], text[n:]


def evaluate_metrics(model, data, tok, block_size, batch_size, iters, device, seed):
    """Estimate token loss and bits per UTF-8 byte over sampled target tokens."""
    if iters < 1 or batch_size < 1 or block_size < 1 or len(data) <= block_size:
        raise ValueError("evaluation sizes must be positive and data longer than block_size")
    byte_lengths = torch.tensor(
        [len(tok.decode([i]).encode("utf-8")) for i in range(tok.vocab_size)],
        dtype=torch.long, device=device,
    )
    generator = torch.Generator().manual_seed(seed)
    total_nats, total_bytes, total_tokens = 0.0, 0, 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(iters):
                x, y = get_batch(data, block_size, batch_size, device, generator=generator)
                logits, _ = model(x)
                total_nats += F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
                ).item()
                total_bytes += byte_lengths[y].sum().item()
                total_tokens += y.numel()
    finally:
        model.train(was_training)
    if total_bytes == 0:
        raise ValueError("sampled targets contain no bytes")
    return {
        "loss_nats_per_token": total_nats / total_tokens,
        "bits_per_byte": total_nats / (math.log(2) * total_bytes),
        "target_tokens": total_tokens,
        "target_bytes": total_bytes,
    }


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


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
    history = []
    training_seconds = 0.0
    model.train()
    synchronize(device)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train, args.block_size, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if step % args.eval_interval == 0 or step == args.steps:
            synchronize(device)
            training_seconds += time.perf_counter() - start
            train_metrics = evaluate_metrics(model, train, tok, args.block_size,
                                             args.batch_size, args.eval_iters, device, 101)
            val_metrics = evaluate_metrics(model, val, tok, args.block_size,
                                           args.batch_size, args.eval_iters, device, 102)
            history.append({"step": step, "train": train_metrics, "val": val_metrics})
            synchronize(device)
            start = time.perf_counter()
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
        "train_loss_nats_per_token": history[-1]["train"]["loss_nats_per_token"],
        "val_loss_nats_per_token": history[-1]["val"]["loss_nats_per_token"],
        "history": history,
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
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--sample-tokens", type=int, default=40)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.block_size, args.n_embd, args.n_head,
           args.n_layer, args.vocab_size, args.bpe_max_chars, args.eval_iters,
           args.eval_interval) < 1:
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
