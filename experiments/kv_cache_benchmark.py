"""Benchmark cached vs uncached greedy decoding on the same model.

    python -m experiments.kv_cache_benchmark
    python -m experiments.kv_cache_benchmark --prompt-tokens 64 --tokens 64
    python -m experiments.kv_cache_benchmark --ckpt checkpoints/tinygpt.pt --tokens 16

Both paths use the same weights, prompt, device, dtype and number of generated
tokens. The cached path is only faster while the whole sequence fits in the
model's context window; past that point the cache is rebuilt from the retained
window (see model/model.py), so this script rejects workloads beyond it.
"""
import argparse
import statistics
import time

import torch

from inference.generate import load_model
from model.model import TinyGPT
from utils import get_device, set_seed


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _sample(logits):
    """Greedy next token, shape (B, 1)."""
    return logits[:, -1].argmax(dim=-1, keepdim=True)


def _run_once(model, idx, new_tokens, use_cache, device):
    """Return (prefill_s, decode_s, output, cache)."""
    _sync(device)
    start = time.perf_counter()
    if use_cache:
        logits, _, past = model(idx, use_cache=True)
    else:
        logits, _ = model(idx)
        past = None
    out = torch.cat([idx, _sample(logits)], dim=1)
    _sync(device)
    prefill = time.perf_counter() - start

    _sync(device)
    start = time.perf_counter()
    for _ in range(new_tokens - 1):
        if use_cache:
            logits, _, past = model(
                out[:, -1:], past_key_values=past, use_cache=True
            )
        else:
            logits, _ = model(out[:, -model.block_size:])
        out = torch.cat([out, _sample(logits)], dim=1)
    _sync(device)
    decode = time.perf_counter() - start
    return prefill, decode, out, past


def _benchmark(model, idx, new_tokens, use_cache, device, warmup, runs):
    for _ in range(warmup):
        _run_once(model, idx, new_tokens, use_cache, device)

    samples = []
    peak = None
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        prefill, decode, out, past = _run_once(
            model, idx, new_tokens, use_cache, device
        )
        if device.type == "cuda":
            used = torch.cuda.max_memory_allocated(device)
            peak = used if peak is None else max(peak, used)
        samples.append((prefill, decode, out))
    prefill = statistics.median(s[0] for s in samples)
    decode = statistics.median(s[1] for s in samples)
    return {
        "prefill": prefill,
        "decode": decode,
        "total": prefill + decode,
        "prefill_tps": idx.size(1) / prefill if prefill else float("inf"),
        "decode_tps": new_tokens / decode if decode else float("inf"),
        "total_tps": new_tokens / (prefill + decode),
        "out": samples[-1][2],
        "cache": past,
        "peak": peak,
    }


def _cache_bytes(past):
    if past is None:
        return 0
    return sum(t.numel() * t.element_size() for kv in past for t in kv)


def _cache_estimate(model, prompt_tokens, new_tokens, batch, dtype):
    layers = len(model.blocks)
    heads = len(model.blocks[0].sa.heads)
    head_size = model.embed.token_embedding.embedding_dim // heads
    cached_tokens = prompt_tokens + new_tokens - 1
    return 2 * layers * batch * heads * cached_tokens * head_size * dtype.itemsize


def _build_model(args, device):
    if args.ckpt:
        model, _tok = load_model(args.ckpt, device)
        return model
    set_seed(args.seed)
    return TinyGPT(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=0.0,
    ).to(device).eval()


def _row(name, uncached, cached, unit=""):
    return f"{name:<24} {uncached:>12.3f}{unit} {cached:>12.3f}{unit}"


def main():
    p = argparse.ArgumentParser(description="Benchmark KV-cached generation.")
    p.add_argument("--ckpt", default=None, help="checkpoint to benchmark (default: random model)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--prompt-tokens", type=int, default=32)
    p.add_argument("--tokens", type=int, default=32, help="tokens to generate")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--runs", type=int, default=5, help="timed repetitions (median reported)")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=65)
    args = p.parse_args()

    if args.prompt_tokens < 1:
        raise SystemExit("error: --prompt-tokens must be >= 1")
    if args.tokens < 1:
        raise SystemExit("error: --tokens must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("error: --batch-size must be >= 1")
    if args.runs < 1:
        raise SystemExit("error: --runs must be >= 1")

    device = get_device(args.device)
    model = _build_model(args, device)
    if args.prompt_tokens > model.block_size:
        raise SystemExit(
            f"error: prompt ({args.prompt_tokens}) exceeds block_size "
            f"({model.block_size})"
        )
    if args.prompt_tokens + args.tokens > model.block_size:
        raise SystemExit(
            f"error: prompt + generated ({args.prompt_tokens + args.tokens}) exceeds "
            f"block_size ({model.block_size}); beyond the window the cache is "
            "rebuilt every step, so there is no decode speedup to measure. "
            "Reduce --tokens or use a model with a larger block_size."
        )

    dtype = next(model.parameters()).dtype
    set_seed(args.seed)
    idx = torch.randint(
        0,
        model.embed.token_embedding.num_embeddings,
        (args.batch_size, args.prompt_tokens),
        device=device,
    )

    uncached = _benchmark(
        model, idx, args.tokens, False, device, args.warmup, args.runs
    )
    cached = _benchmark(
        model, idx, args.tokens, True, device, args.warmup, args.runs
    )

    match = torch.equal(uncached["out"], cached["out"])
    speedup = uncached["total"] / cached["total"]
    cache_actual = _cache_bytes(cached["cache"])
    cache_est = _cache_estimate(
        model, args.prompt_tokens, args.tokens, args.batch_size, dtype
    )

    print(f"Device: {device} | dtype {dtype} | torch {torch.__version__}")
    print(
        f"Model: n_layer={len(model.blocks)} "
        f"n_head={len(model.blocks[0].sa.heads)} "
        f"n_embd={model.embed.token_embedding.embedding_dim} "
        f"block_size={model.block_size}"
    )
    print(
        f"Prompt tokens: {args.prompt_tokens} | Generated tokens: {args.tokens} | "
        f"Batch: {args.batch_size} | Runs: {args.runs} (median)"
    )
    print()
    print(f"{'':<24} {'Uncached':>13} {'Cached':>13}")
    print(_row("prefill time (s)", uncached["prefill"], cached["prefill"]))
    print(_row("decode time (s)", uncached["decode"], cached["decode"]))
    print(_row("end-to-end (s)", uncached["total"], cached["total"]))
    print(_row("prefill tokens/s", uncached["prefill_tps"], cached["prefill_tps"]))
    print(_row("decode tokens/s", uncached["decode_tps"], cached["decode_tps"]))
    print(_row("end-to-end tokens/s", uncached["total_tps"], cached["total_tps"]))
    peak_u = "n/a" if uncached["peak"] is None else f"{uncached['peak'] / 2**20:.2f}"
    peak_c = "n/a" if cached["peak"] is None else f"{cached['peak'] / 2**20:.2f}"
    print(f"{'peak memory (MiB)':<24} {peak_u:>13} {peak_c:>13}")
    print()
    print(f"speedup (uncached / cached end-to-end): {speedup:.2f}x")
    print(
        f"KV cache at final step: {cache_actual / 2**20:.2f} MiB "
        f"(formula: {cache_est / 2**20:.2f} MiB, "
        f"{args.prompt_tokens + args.tokens - 1} cached tokens)"
    )
    print(f"greedy output match: {'yes' if match else 'NO'}")
    if not match:
        raise SystemExit("error: cached and uncached outputs diverged")


if __name__ == "__main__":
    main()
