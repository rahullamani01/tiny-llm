"""Deterministically evaluate assistant-token loss on complete conversations."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from chat.template import CHAT_SPECIAL_TOKENS, IGNORE_INDEX, build_sft_example
from inference.generate import load_model
from training.sft import load_conversations
from utils import get_device


def evaluate_conversations(model, tok, conversations, device, system=None):
    """Score each assistant target once in independent, non-overlapping chunks.

    Conversation and chunk boundaries reset context. Targets are weighted by
    token count, including the assistant's closing end token.
    """
    missing = [t for t in CHAT_SPECIAL_TOKENS if t not in tok.special_tokens]
    if missing:
        raise ValueError("checkpoint is missing chat special tokens")
    total_loss, total_tokens, scored_conversations = 0.0, 0, 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for conv in conversations:
                ids, labels = build_sft_example(
                    conv["messages"], tok, system=conv.get("system") or system
                )
                conversation_tokens = 0
                for start in range(0, len(ids) - 1, model.block_size):
                    inputs = ids[start:start + model.block_size]
                    targets = labels[start + 1:start + 1 + len(inputs)]
                    inputs = inputs[:len(targets)]
                    count = sum(target != IGNORE_INDEX for target in targets)
                    if not count:
                        continue
                    x = torch.tensor([inputs], dtype=torch.long, device=device)
                    y = torch.tensor(targets, dtype=torch.long, device=device)
                    logits, _ = model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y, reduction="sum"
                    )
                    if not torch.isfinite(loss):
                        raise ValueError("non-finite evaluation loss")
                    total_loss += loss.item()
                    total_tokens += count
                    conversation_tokens += count
                scored_conversations += int(conversation_tokens > 0)
    finally:
        model.train(was_training)
    if not total_tokens:
        raise ValueError("no supervised assistant targets in the input conversations")
    mean = total_loss / total_tokens
    return {
        "conversations": len(conversations),
        "scored_conversations": scored_conversations,
        "assistant_target_tokens": total_tokens,
        "loss_nats_per_token": mean,
        "perplexity": math.exp(mean) if mean < 709 else None,
    }


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="trusted SFT checkpoint")
    p.add_argument("--data", required=True, help="held-out conversation JSONL")
    p.add_argument("--system", default=None, help="fallback system prompt")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--output", default=None, help="new JSON report path")
    args = p.parse_args()
    if args.output and Path(args.output).exists():
        p.error("output already exists")
    try:
        device = get_device(args.device)
        model, tok = load_model(args.ckpt, device)
        conversations = load_conversations(args.data)
        metrics = evaluate_conversations(model, tok, conversations, device, args.system)
        report = {
            "checkpoint_sha256": file_hash(args.ckpt),
            "data_sha256": file_hash(args.data),
            "device": str(device), "torch_version": torch.__version__,
            "context_tokens": model.block_size, "fallback_system": args.system,
            "method": "conversation_separated_nonoverlapping_chunks_v1",
            "metrics": metrics,
        }
        text = json.dumps(report, indent=2) + "\n"
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as f:
                f.write(text)
        print(text, end="")
    except (OSError, KeyError, ValueError) as e:
        raise SystemExit(f"error: {e}") from None


if __name__ == "__main__":
    main()
