import argparse
import json
import os

import torch

from checkpoint import load_checkpoint, save_checkpoint
from model.model import TinyGPT
from tokenizer import BPETokenizer, CharTokenizer, load_tokenizer, save_tokenizer
from training.dataset import get_batch, split_data
from training.evaluate import evaluate_loss
from training.schedule import get_lr
from utils import get_device, set_seed

# ---- hyperparameters ----
CFG = dict(
    batch_size=32,
    block_size=64,      # context length
    n_embd=128,         # embedding size
    n_head=4,           # attention heads (n_embd must divide by n_head)
    n_layer=4,          # transformer blocks
    dropout=0.1,
    lr=1e-3,
    min_lr=1e-4,        # cosine decay floor
    warmup_steps=100,   # linear warmup
    lr_schedule="cosine",
    grad_clip=1.0,
    max_steps=5000,
    eval_interval=500,
    eval_iters=50,
    val_fraction=0.1,
    seed=1337,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train TinyGPT.")
    p.add_argument("--data", default="data/training.txt")
    p.add_argument("--out", default="checkpoints/tinygpt.pt", help="latest/resumable checkpoint")
    p.add_argument("--best-out", default=None, help="best-val checkpoint (default: next to --out)")
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    p.add_argument("--vocab-size", type=int, default=512,
                   help="BPE vocabulary size, including base characters")
    p.add_argument("--bpe-max-chars", type=int, default=100_000,
                   help="text prefix used to learn BPE merges")
    p.add_argument("--vocab-file", default=None,
                   help="BPE vocabulary JSON (default: next to --out)")
    p.add_argument("--resume", action="store_true", help="continue from --out")
    p.add_argument("--tie-weights", action="store_true", help="share token embedding and output weights")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=CFG["seed"])
    p.add_argument("--batch-size", type=int, default=CFG["batch_size"])
    p.add_argument("--block-size", type=int, default=CFG["block_size"])
    p.add_argument("--n-embd", type=int, default=CFG["n_embd"])
    p.add_argument("--n-head", type=int, default=CFG["n_head"])
    p.add_argument("--n-layer", type=int, default=CFG["n_layer"])
    p.add_argument("--dropout", type=float, default=CFG["dropout"])
    p.add_argument("--lr", type=float, default=CFG["lr"])
    p.add_argument("--min-lr", type=float, default=CFG["min_lr"])
    p.add_argument("--warmup-steps", type=int, default=CFG["warmup_steps"])
    p.add_argument("--lr-schedule", choices=["cosine", "none"], default=CFG["lr_schedule"])
    p.add_argument("--grad-clip", type=float, default=CFG["grad_clip"], help="0 disables clipping")
    p.add_argument("--max-steps", type=int, default=CFG["max_steps"])
    p.add_argument("--eval-interval", type=int, default=CFG["eval_interval"])
    p.add_argument("--eval-iters", type=int, default=CFG["eval_iters"])
    p.add_argument("--val-fraction", type=float, default=CFG["val_fraction"])
    p.add_argument("--sample-tokens", type=int, default=500, help="0 skips the final sample")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def validate_config(cfg):
    if cfg["max_steps"] < 1:
        raise ValueError("max_steps must be >= 1")
    if cfg["batch_size"] < 1:
        raise ValueError("batch_size must be >= 1")
    if cfg["block_size"] < 2:
        raise ValueError("block_size must be >= 2")
    if cfg["n_embd"] < 1 or cfg["n_head"] < 1:
        raise ValueError("n_embd and n_head must be >= 1")
    if cfg["n_embd"] % cfg["n_head"] != 0:
        raise ValueError("n_embd must be divisible by n_head")
    if cfg["n_layer"] < 1:
        raise ValueError("n_layer must be >= 1")
    if not 0.0 <= cfg["dropout"] < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if cfg["lr"] < 0 or cfg["min_lr"] < 0 or cfg["min_lr"] > cfg["lr"]:
        raise ValueError("require 0 <= min_lr <= lr")
    if cfg["warmup_steps"] < 0 or cfg["warmup_steps"] > cfg["max_steps"]:
        raise ValueError("require 0 <= warmup_steps <= max_steps")
    if cfg["grad_clip"] < 0:
        raise ValueError("grad_clip must be >= 0")
    if cfg["eval_interval"] < 1 or cfg["eval_iters"] < 1:
        raise ValueError("eval_interval and eval_iters must be >= 1")
    if not 0.0 < cfg["val_fraction"] < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    if cfg["lr_schedule"] not in ("cosine", "none"):
        raise ValueError("lr_schedule must be 'cosine' or 'none'")


def _checkpoint_matches(ckpt_cfg, model_cfg):
    """Raise SystemExit if a checkpoint's architecture differs from the current one."""
    keys = ("vocab_size", "block_size", "n_embd", "n_head", "n_layer", "dropout")
    for key in keys:
        old = ckpt_cfg.get(key)
        new = model_cfg[key]
        if old != new:
            raise SystemExit(
                f"error: checkpoint {key}={old!r} does not match current {key}={new!r}; "
                "pass the original architecture or start a new run"
            )
    if ckpt_cfg.get("tie_weights", False) != model_cfg["tie_weights"]:
        raise SystemExit(
            "error: checkpoint tie_weights does not match; pass --tie-weights if the "
            "checkpoint was trained with it"
        )


def build_tokenizer(args, text):
    """Build a char tokenizer, or load/train a BPE vocabulary."""
    if args.tokenizer == "char":
        return CharTokenizer(text), None

    if args.vocab_size < 1:
        raise SystemExit("error: --vocab-size must be >= 1")
    if args.bpe_max_chars < 1:
        raise SystemExit("error: --bpe-max-chars must be >= 1")

    path = args.vocab_file or os.path.join(
        os.path.dirname(args.out) or ".", "bpe.json"
    )
    if os.path.exists(path):
        try:
            tok = load_tokenizer(path)
        except (OSError, ValueError, KeyError) as e:
            raise SystemExit(f"error: could not load BPE vocabulary {path}: {e}")
        if not isinstance(tok, BPETokenizer):
            raise SystemExit(f"error: {path} does not contain a BPE vocabulary")
        missing = tok.missing_chars(text)
        if missing:
            shown = ", ".join(repr(c) for c in missing)
            raise SystemExit(f"error: BPE vocabulary is missing data characters: {shown}")
        print(f"loaded BPE vocabulary: {tok.vocab_size} tokens from {path}")
        return tok, path

    print(f"training BPE vocabulary (target {args.vocab_size} tokens)...")
    try:
        tok = BPETokenizer.train(
            text,
            args.vocab_size,
            max_chars=args.bpe_max_chars,
        )
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    save_tokenizer(tok, path)
    print(f"saved BPE vocabulary: {path}")
    return tok, path


def main():
    args = parse_args()
    cfg = {key: getattr(args, key, value) for key, value in CFG.items()}

    if args.sample_tokens < 0:
        raise SystemExit("error: --sample-tokens must be >= 0")

    try:
        validate_config(cfg)
        device = get_device(args.device)
    except ValueError as e:
        raise SystemExit(f"error: {e}")

    set_seed(cfg["seed"])
    print("device:", device)

    # ---- data ----
    try:
        text = open(args.data, encoding="utf-8").read()
    except FileNotFoundError:
        raise SystemExit(
            f"error: data file not found: {args.data}\nrun: python data/download.py"
        )

    tok, vocab_path = build_tokenizer(args, text)
    tok_state = tok.to_dict()
    print(
        f"tokenizer: {args.tokenizer} | vocab {tok.vocab_size} | "
        f"chars {len(tok.chars)}"
    )
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    try:
        splits = split_data(data, cfg["val_fraction"])
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    for name, d in splits.items():
        if len(d) <= cfg["block_size"]:
            raise SystemExit(
                f"error: {name} split has {len(d)} tokens; need more than "
                f"block_size={cfg['block_size']}"
            )

    # ---- model ----
    model_cfg = dict(
        vocab_size=tok.vocab_size,
        block_size=cfg["block_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        dropout=cfg["dropout"],
        tie_weights=args.tie_weights,
    )
    model = TinyGPT(**model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(
        f"tokens: {len(data)} | train {len(splits['train'])} | "
        f"val {len(splits['val'])}"
    )

    best_path = args.best_out or os.path.join(os.path.dirname(args.out) or ".", "best.pt")
    if os.path.abspath(best_path) == os.path.abspath(args.out):
        raise SystemExit("error: --best-out must differ from --out")
    history_path = os.path.join(os.path.dirname(args.out) or ".", "history.json")

    # ---- optional resume ----
    start_step = 0
    history = []
    best_val = float("inf")
    if args.resume:
        if not os.path.exists(args.out):
            raise SystemExit(f"error: no checkpoint at {args.out} to resume from")
        ckpt = load_checkpoint(args.out, device)
        ckpt_tok_state = ckpt.get("tokenizer")
        if ckpt_tok_state is not None:
            if ckpt_tok_state != tok_state:
                kind = ckpt_tok_state.get("type", "unknown")
                raise SystemExit(
                    f"error: checkpoint uses a {kind!r} tokenizer that does not match "
                    f"the current {args.tokenizer!r} tokenizer; pass the original "
                    "--tokenizer/--vocab-file"
                )
        elif list(ckpt["chars"]) != tok.chars:
            raise SystemExit("error: checkpoint vocabulary does not match the data file")
        _checkpoint_matches(ckpt["config"], model_cfg)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        history = list(ckpt.get("history", []))
        best_val = float(ckpt.get("best_val_loss", float("inf")))
        if start_step > cfg["max_steps"]:
            raise SystemExit(
                f"error: checkpoint is at step {start_step}; "
                f"pass --max-steps greater than {start_step}"
            )
        if "step" not in ckpt:
            print("warning: legacy checkpoint has no step/optimizer; starting at step 0")
        print(f"resumed from {args.out} at step {start_step}")

    train_config = dict(cfg)
    train_config.update(
        data=args.data,
        device=str(device),
        tie_weights=args.tie_weights,
        tokenizer=args.tokenizer,
        vocab_file=vocab_path,
        vocab_size=tok.vocab_size,
    )

    def current_lr(step):
        return get_lr(
            step,
            cfg["max_steps"],
            cfg["lr"],
            cfg["lr_schedule"],
            cfg["warmup_steps"],
            cfg["min_lr"],
        )

    def estimate_loss():
        model.eval()
        out = {}
        for name, d in splits.items():
            out[name] = evaluate_loss(
                model,
                d,
                cfg["block_size"],
                cfg["batch_size"],
                cfg["eval_iters"],
                device,
            )
        model.train()
        return out

    def save(step):
        save_checkpoint(
            args.out,
            model,
            opt,
            model_cfg,
            tok.chars,
            step,
            history,
            best_val,
            train_config,
            tokenizer=tok_state,
        )
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    def evaluate_and_log(step):
        nonlocal best_val
        metrics = estimate_loss()
        entry = (step, metrics["train"], metrics["val"])
        if history and history[-1][0] == step:
            history[-1] = entry
        else:
            history.append(entry)

        improved = metrics["val"] < best_val
        if improved:
            best_val = metrics["val"]
        print(
            f"step {step}: train {metrics['train']:.4f}, "
            f"val {metrics['val']:.4f}, lr {current_lr(step):.2e}, "
            f"best {best_val:.4f}"
        )
        save(step)
        if improved:
            save_checkpoint(
                best_path,
                model,
                opt,
                model_cfg,
                tok.chars,
                step,
                history,
                best_val,
                train_config,
                tokenizer=tok_state,
            )
            print(f"  new best -> {best_path}")
        return metrics

    # ---- training loop ----
    model.train()
    for step in range(start_step, cfg["max_steps"]):
        if step % cfg["eval_interval"] == 0:
            evaluate_and_log(step)

        lr = current_lr(step)
        for group in opt.param_groups:
            group["lr"] = lr

        xb, yb = get_batch(
            splits["train"], cfg["block_size"], cfg["batch_size"], device
        )
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()

    evaluate_and_log(cfg["max_steps"])

    # ---- loss curve ----
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping loss curve")
        else:
            steps = [h[0] for h in history]
            plt.figure(figsize=(6, 4))
            plt.plot(steps, [h[1] for h in history], label="train")
            plt.plot(steps, [h[2] for h in history], label="val")
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.legend()
            plt.title("TinyGPT training")
            os.makedirs("assets", exist_ok=True)
            plt.savefig("assets/loss_curve.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("saved assets/loss_curve.png")

    # ---- sample ----
    if args.sample_tokens > 0:
        model.eval()
        start = torch.tensor([[tok.stoi.get("\n", 0)]], dtype=torch.long, device=device)
        out = model.generate(start, args.sample_tokens, temperature=0.8, top_k=40)
        print("\n--- sample ---")
        print(tok.decode(out[0].tolist()))

    print(f"saved {args.out}")
    print(f"best {best_path} (val loss {best_val:.4f})")


if __name__ == "__main__":
    main()
