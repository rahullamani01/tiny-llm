"""Supervised fine-tuning (SFT) for chat.

Expects JSONL conversations, one per line:

    {"system": "You are helpful.", "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"}
    ]}

Only assistant content and the closing ``<|end|>`` token contribute to the loss.
Run with ``--init-from`` to start from a pretrained Base checkpoint; the token
embedding and output head are resized automatically when chat special tokens are
added.
"""
import argparse
import json
import os
import random

import torch
import torch.nn as nn

from chat.template import CHAT_SPECIAL_TOKENS, build_sft_example, normalize_messages
from checkpoint import load_checkpoint, save_checkpoint
from model.model import TinyGPT
from tokenizer import (
    BPETokenizer,
    ByteTokenizer,
    CharTokenizer,
    load_tokenizer,
    save_tokenizer,
    tokenizer_from_state,
    with_special_tokens,
)
from training.train import CFG, _checkpoint_matches, run_optimizer_step, validate_config
from utils import get_device, set_seed

IGNORE_INDEX = -100


def load_conversations(path):
    """Read JSONL conversations and validate their message structure."""
    conversations = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from None

            if isinstance(obj, list):
                messages, system = obj, None
            elif isinstance(obj, dict):
                messages = obj.get("messages") or obj.get("conversation")
                system = obj.get("system")
            else:
                raise ValueError(f"{path}:{line_no}: expected an object or list")

            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_no}: 'messages' must be a non-empty list")
            try:
                normalize_messages(messages, system=system)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{path}:{line_no}: {e}") from None
            conversations.append({"messages": messages, "system": system})

    if not conversations:
        raise ValueError(f"{path}: no conversations found")
    return conversations


def conversation_text(conversations, system=None):
    """Plain text used to learn a tokenizer (special tokens are added later)."""
    chunks = []
    for conv in conversations:
        prompt = conv.get("system") or system
        if prompt:
            chunks.append(prompt)
        for message in conv["messages"]:
            chunks.append(message["content"] if isinstance(message, dict) else message.content)
    return "\n".join(chunks)


def prompt_key(conversation):
    """Group exact normalized user prompts, including multi-turn user history."""
    return tuple(
        " ".join(message["content"].split()).casefold()
        for message in conversation["messages"] if message["role"] == "user"
    )


def split_conversations(conversations, val_fraction=0.1, seed=0):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    if len(conversations) < 2:
        raise ValueError("need at least 2 conversations to make train/val splits")

    groups = {}
    for conv in conversations:
        groups.setdefault(prompt_key(conv), []).append(conv)
    keys = list(groups)
    if len(keys) < 2:
        raise ValueError("need at least 2 distinct user prompt groups for train/val")
    random.Random(seed).shuffle(keys)
    n_train = min(max(int((1.0 - val_fraction) * len(keys)), 1), len(keys) - 1)
    train = [conv for key in keys[:n_train] for conv in groups[key]]
    val = [conv for key in keys[n_train:] for conv in groups[key]]
    return train, val


def encode_conversations(conversations, tokenizer, system=None):
    """Concatenate conversations into aligned ``(ids, labels)`` tensors."""
    ids, labels = [], []
    for conv in conversations:
        prompt = conv.get("system") or system
        conv_ids, conv_labels = build_sft_example(
            conv["messages"], tokenizer, system=prompt
        )
        ids.extend(conv_ids)
        labels.extend(conv_labels)
    if len(ids) != len(labels):
        raise ValueError("ids and labels must have the same length")
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def _window_with_labels(labels, block_size, generator=None):
    """Pick a start index whose window contains at least one supervised token."""
    valid = torch.nonzero(labels[1:] != IGNORE_INDEX).flatten() + 1
    if valid.numel() == 0:
        raise ValueError("conversation split has no assistant tokens")
    pos = int(valid[torch.randint(valid.numel(), (1,), generator=generator)].item())
    return min(max(0, pos - 1), len(labels) - block_size - 1)


def get_sft_batch(ids, labels, block_size, batch_size, device, generator=None):
    """Sample fixed-size windows; every row has at least one trained token."""
    if len(ids) != len(labels):
        raise ValueError("ids and labels must have the same length")
    if len(ids) <= block_size:
        raise ValueError("conversation data is shorter than block_size + 1 tokens")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    starts = torch.randint(len(ids) - block_size, (batch_size,), generator=generator)
    x = torch.stack([ids[i : i + block_size] for i in starts])
    y = torch.stack([labels[i + 1 : i + block_size + 1] for i in starts])

    for row in range(batch_size):
        if (y[row] != IGNORE_INDEX).any():
            continue
        start = _window_with_labels(labels, block_size, generator=generator)
        x[row] = ids[start : start + block_size]
        y[row] = labels[start + 1 : start + block_size + 1]
    return x.to(device), y.to(device)


def evaluate_sft_loss(
    model, ids, labels, block_size, batch_size=8, iters=20, device=None, seed=None
):
    """Average batch loss over assistant tokens, preserving the model's mode.

    When seeded, batch sampling uses a local generator instead of changing the
    training random state. Reusing the seed evaluates the same sampled windows.
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")
    device = device or next(model.parameters()).device
    generator = None if seed is None else torch.Generator().manual_seed(seed)

    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        with torch.no_grad():
            for _ in range(iters):
                xb, yb = get_sft_batch(
                    ids, labels, block_size, batch_size, device, generator=generator
                )
                _, loss = model(xb, yb)
                if torch.isfinite(loss):
                    total += loss.item()
                    count += 1
    finally:
        model.train(was_training)
    return total / count if count else float("inf")


def _resize_embeddings(model, vocab_size):
    """Grow token embeddings/head in place, keeping old rows and init new rows."""
    old_embedding = model.embed.token_embedding
    if old_embedding.num_embeddings == vocab_size:
        return

    if vocab_size < old_embedding.num_embeddings:
        raise ValueError("the new vocabulary cannot be smaller than the checkpoint's")

    tied = model.lm_head.weight.data_ptr() == old_embedding.weight.data_ptr()
    new_embedding = nn.Embedding(vocab_size, old_embedding.embedding_dim)
    nn.init.normal_(new_embedding.weight, mean=0.0, std=0.02)
    with torch.no_grad():
        new_embedding.weight[: old_embedding.num_embeddings].copy_(old_embedding.weight)

    old_head = model.lm_head
    new_head = nn.Linear(old_head.in_features, vocab_size, bias=old_head.bias is not None)
    nn.init.normal_(new_head.weight, mean=0.0, std=0.02)
    with torch.no_grad():
        new_head.weight[: old_head.out_features].copy_(old_head.weight)
        if old_head.bias is not None:
            new_head.bias[: old_head.out_features].copy_(old_head.bias)

    model.embed.token_embedding = new_embedding
    model.lm_head = new_head
    if tied:
        model.lm_head.weight = model.embed.token_embedding.weight


def load_base_weights(model, path, device):
    """Load a pretrained Base checkpoint into a model with a resized vocabulary."""
    ckpt = load_checkpoint(path, device)
    base_cfg = ckpt["config"]
    current = {
        "block_size": model.block_size,
        "n_embd": model.embed.token_embedding.embedding_dim,
        "n_head": len(model.blocks[0].sa.heads),
        "n_layer": len(model.blocks),
    }
    for key, value in current.items():
        if base_cfg.get(key) != value:
            raise SystemExit(
                f"error: base checkpoint {key}={base_cfg.get(key)!r} does not match "
                f"the SFT model {key}={value!r}; pass the same architecture flags"
            )
    if base_cfg.get("pos_embedding", "learned") != model.pos_embedding:
        raise SystemExit("error: base checkpoint uses a different position scheme")
    tied_now = (
        model.lm_head.weight.data_ptr()
        == model.embed.token_embedding.weight.data_ptr()
    )
    if bool(base_cfg.get("tie_weights", False)) != tied_now:
        raise SystemExit(
            "error: base checkpoint tie_weights does not match; pass --tie-weights "
            "if the Base run used it"
        )

    state = dict(ckpt["model"])
    old_embedding = state.pop("embed.token_embedding.weight", None)
    old_head = state.pop("lm_head.weight", None)
    old_head_bias = state.pop("lm_head.bias", None)
    if old_embedding is None:
        raise SystemExit("error: base checkpoint has no token embedding")

    with torch.no_grad():
        n = min(old_embedding.shape[0], model.embed.token_embedding.num_embeddings)
        model.embed.token_embedding.weight[:n].copy_(old_embedding[:n])
        if (
            old_head is not None
            and model.lm_head.weight.data_ptr()
            != model.embed.token_embedding.weight.data_ptr()
        ):
            n = min(old_head.shape[0], model.lm_head.out_features)
            model.lm_head.weight[:n].copy_(old_head[:n])
        if old_head_bias is not None and model.lm_head.bias is not None:
            n = min(old_head_bias.shape[0], model.lm_head.out_features)
            model.lm_head.bias[:n].copy_(old_head_bias[:n])

    missing, unexpected = model.load_state_dict(state, strict=False)
    handled = {"embed.token_embedding.weight", "lm_head.weight", "lm_head.bias"}
    missing = [key for key in missing if key not in handled]
    if missing or unexpected:
        raise SystemExit(
            f"error: could not load base checkpoint "
            f"(missing={sorted(missing)}, unexpected={sorted(unexpected)})"
        )
    return ckpt


def build_tokenizer(args, conversations, base_ckpt=None):
    """Load or train a tokenizer, then append chat special tokens."""
    if args.vocab_file and os.path.exists(args.vocab_file):
        tok = load_tokenizer(args.vocab_file)
        return with_special_tokens(tok, CHAT_SPECIAL_TOKENS), args.vocab_file

    if base_ckpt is not None:
        state = base_ckpt.get("tokenizer")
        if state is not None:
            tok = tokenizer_from_state(state)
        else:
            tok = CharTokenizer.from_chars(base_ckpt["chars"])
        return with_special_tokens(tok, CHAT_SPECIAL_TOKENS), args.vocab_file

    text = conversation_text(conversations, system=args.system)
    if args.tokenizer in ("char", "byte"):
        tok = CharTokenizer(text) if args.tokenizer == "char" else ByteTokenizer()
        specials = with_special_tokens(tok, CHAT_SPECIAL_TOKENS)
        path = args.vocab_file or os.path.join(
            os.path.dirname(args.out) or ".", "chat-vocab.json"
        )
        save_tokenizer(specials, path)
        return specials, path

    path = args.vocab_file or os.path.join(
        os.path.dirname(args.out) or ".", "chat-vocab.json"
    )
    tok = BPETokenizer.train(
        text,
        vocab_size=args.vocab_size,
        max_chars=args.bpe_max_chars,
    )
    specials = with_special_tokens(tok, CHAT_SPECIAL_TOKENS)
    save_tokenizer(specials, path)
    print(f"saved chat vocabulary: {path}")
    return specials, path


def parse_args():
    p = argparse.ArgumentParser(description="Supervised fine-tuning for chat.")
    p.add_argument("--data", required=True, help="JSONL conversations")
    p.add_argument("--val-data", default=None, help="separate held-out JSONL conversations")
    p.add_argument("--out", default="checkpoints/tinygpt-chat.pt")
    p.add_argument("--best-out", default=None)
    p.add_argument("--init-from", default=None, help="pretrained Base checkpoint")
    p.add_argument("--resume", action="store_true", help="continue from --out")
    p.add_argument("--tokenizer", choices=["char", "bpe", "byte"], default="bpe")
    p.add_argument("--vocab-file", default=None)
    p.add_argument("--vocab-size", type=int, default=2048)
    p.add_argument("--bpe-max-chars", type=int, default=200_000)
    p.add_argument("--system", default=None, help="default system prompt")
    p.add_argument("--tie-weights", action="store_true")
    p.add_argument("--pos-embedding", choices=["learned", "rope"], default="learned")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=CFG["seed"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-embd", type=int, default=CFG["n_embd"])
    p.add_argument("--n-head", type=int, default=CFG["n_head"])
    p.add_argument("--n-layer", type=int, default=CFG["n_layer"])
    p.add_argument("--dropout", type=float, default=CFG["dropout"])
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr", type=float, default=CFG["min_lr"])
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--lr-schedule", choices=["cosine", "none"], default=CFG["lr_schedule"])
    p.add_argument("--weight-decay", type=float, default=CFG["weight_decay"])
    p.add_argument("--grad-clip", type=float, default=CFG["grad_clip"])
    p.add_argument("--grad-accum-steps", type=int, default=CFG["grad_accum_steps"])
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--eval-seed", type=int, default=0,
                   help="fixed sampling seed for evaluation batches")
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = {key: getattr(args, key, default) for key, default in CFG.items()}
    if args.init_from and args.resume:
        raise SystemExit("error: --init-from and --resume are mutually exclusive")

    try:
        validate_config(cfg)
        device = get_device(args.device)
    except ValueError as e:
        raise SystemExit(f"error: {e}")

    if args.vocab_size < 1 or args.bpe_max_chars < 1:
        raise SystemExit("error: --vocab-size and --bpe-max-chars must be >= 1")

    set_seed(cfg["seed"])
    print("device:", device)

    try:
        conversations = load_conversations(args.data)
    except (OSError, ValueError) as e:
        raise SystemExit(f"error: {e}")
    try:
        if args.val_data:
            train_convs = conversations
            val_convs = load_conversations(args.val_data)
            if {prompt_key(c) for c in train_convs} & {prompt_key(c) for c in val_convs}:
                raise ValueError("train and validation contain matching normalized user prompts")
        else:
            train_convs, val_convs = split_conversations(
                conversations, cfg["val_fraction"], seed=cfg["seed"]
            )
    except (OSError, ValueError) as e:
        raise SystemExit(f"error: {e}") from None
    print(f"conversations: train {len(train_convs)} | val {len(val_convs)}")

    base_ckpt = None
    if args.init_from:
        if not os.path.exists(args.init_from):
            raise SystemExit(f"error: no base checkpoint at {args.init_from}")
        base_ckpt = load_checkpoint(args.init_from, device)
    tok, vocab_path = build_tokenizer(args, train_convs, base_ckpt)
    tok_state = tok.to_dict()
    print(
        f"tokenizer: {tok_state.get('type')} | vocab {tok.vocab_size} | "
        f"specials {len(tok.special_tokens)}"
    )

    try:
        train_ids, train_labels = encode_conversations(
            train_convs, tok, system=args.system
        )
        val_ids, val_labels = encode_conversations(val_convs, tok, system=args.system)
    except KeyError as e:
        raise SystemExit(
            f"error: chat data contains a character missing from the tokenizer: {e}"
        ) from None
    except ValueError as e:
        raise SystemExit(f"error: {e}")

    window = cfg["block_size"]
    if len(train_ids) <= window or len(val_ids) <= window:
        window = min(cfg["block_size"], len(train_ids) - 1, len(val_ids) - 1)
        if window < 2:
            raise SystemExit(
                "error: a split is too short to build any training windows; add "
                "more conversations or lower --block-size"
            )
        print(
            f"note: using {window}-token SFT windows because a split is shorter "
            f"than block_size={cfg['block_size']}"
        )

    model_cfg = dict(
        vocab_size=tok.vocab_size,
        block_size=cfg["block_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        dropout=cfg["dropout"],
        tie_weights=args.tie_weights,
        pos_embedding=args.pos_embedding,
    )
    model = TinyGPT(**model_cfg).to(device)
    if base_ckpt is not None:
        _resize_embeddings(model, tok.vocab_size)
        load_base_weights(model, args.init_from, device)
        print(f"initialized from {args.init_from}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(
        f"tokens: train {len(train_ids)} | val {len(val_ids)} | "
        f"block_size {cfg['block_size']}"
    )

    best_path = args.best_out or os.path.join(
        os.path.dirname(args.out) or ".", "chat-best.pt"
    )
    if os.path.abspath(best_path) == os.path.abspath(args.out):
        raise SystemExit("error: --best-out must differ from --out")
    history_path = os.path.join(os.path.dirname(args.out) or ".", "chat-history.json")

    start_step = 0
    history = []
    best_val = float("inf")
    if args.resume:
        if not os.path.exists(args.out):
            raise SystemExit(f"error: no checkpoint at {args.out} to resume from")
        ckpt = load_checkpoint(args.out, device)
        if ckpt.get("train_config", {}).get("sft_objective") != "next_token_v1":
            raise SystemExit(
                "error: cannot resume an older SFT checkpoint trained with the "
                "previous objective; start a new SFT run"
            )
        _checkpoint_matches(ckpt["config"], model_cfg)
        if ckpt.get("tokenizer") is not None and ckpt["tokenizer"] != tok_state:
            raise SystemExit("error: checkpoint tokenizer does not match the current vocabulary")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        history = list(ckpt.get("history", []))
        best_val = float(ckpt.get("best_val_loss", float("inf")))
        if ckpt.get("train_config", {}).get("eval_seed") != args.eval_seed:
            best_val = float("inf")
            print(
                "warning: evaluation sampling changed; resetting best validation "
                "loss (older history uses different batches)"
            )
        print(f"resumed from {args.out} at step {start_step}")

    train_config = dict(cfg)
    train_config.update(
        data=args.data,
        val_data=args.val_data,
        eval_seed=args.eval_seed,
        sft_objective="next_token_v1",
        device=str(device),
        init_from=args.init_from,
        tokenizer=args.tokenizer,
        vocab_file=vocab_path,
        vocab_size=tok.vocab_size,
        system=args.system,
        tie_weights=args.tie_weights,
        pos_embedding=args.pos_embedding,
        sft_block_size=window,
    )

    def current_lr(step):
        from training.schedule import get_lr

        return get_lr(
            step,
            cfg["max_steps"],
            cfg["lr"],
            cfg["lr_schedule"],
            cfg["warmup_steps"],
            cfg["min_lr"],
        )

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

    stale_evals = 0
    stop_training = False

    def evaluate_and_log(step):
        nonlocal best_val, stale_evals, stop_training
        train_loss = evaluate_sft_loss(
            model,
            train_ids,
            train_labels,
            window,
            cfg["batch_size"],
            cfg["eval_iters"],
            device,
            seed=args.eval_seed,
        )
        val_loss = evaluate_sft_loss(
            model,
            val_ids,
            val_labels,
            window,
            cfg["batch_size"],
            cfg["eval_iters"],
            device,
            seed=args.eval_seed + 1,
        )
        entry = (step, train_loss, val_loss)
        if history and history[-1][0] == step:
            history[-1] = entry
        else:
            history.append(entry)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            stale_evals = 0
        else:
            stale_evals += 1
        print(
            f"step {step}: train {train_loss:.4f}, val {val_loss:.4f}, "
            f"lr {current_lr(step):.2e}, best {best_val:.4f}"
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
        if cfg["patience"] > 0 and stale_evals >= cfg["patience"]:
            stop_training = True
            print(f"early stopping: no improvement for {stale_evals} evaluations")

    for step in range(start_step, cfg["max_steps"]):
        if step % cfg["eval_interval"] == 0:
            evaluate_and_log(step)
            if stop_training:
                break

        for group in opt.param_groups:
            group["lr"] = current_lr(step)
        batches = [
            get_sft_batch(
                train_ids,
                train_labels,
                window,
                cfg["batch_size"],
                device,
            )
            for _ in range(cfg["grad_accum_steps"])
        ]
        run_optimizer_step(model, opt, batches, cfg["grad_clip"])

    if not stop_training:
        evaluate_and_log(cfg["max_steps"])
    else:
        print(f"stopped early at step {history[-1][0]}")

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
            plt.title("TinyGPT chat SFT")
            os.makedirs("assets", exist_ok=True)
            plt.savefig("assets/chat_loss_curve.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("saved assets/chat_loss_curve.png")

    print(f"saved {args.out}")
    print(f"best {best_path} (val loss {best_val:.4f})")


if __name__ == "__main__":
    main()
