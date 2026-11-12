"""Interactive streaming chat with an SFT checkpoint.

    python -m inference.chat --ckpt checkpoints/tinygpt-chat.pt

Type a message and the assistant answers. Slash commands change sampling or the
system prompt. Conversation history is trimmed to the model's context window.
"""
import argparse

import torch

from chat.template import CHAT_SPECIAL_TOKENS, build_chat_prompt
from inference.generate import load_model
from utils import get_device, set_seed

DEFAULT_SYSTEM = "You are a helpful, concise assistant."

HELP = """commands:
  /system <text>  replace the system prompt and reset the conversation
  /temp <x>       temperature (0 = greedy)
  /topk <n>       keep the n most likely tokens (0 = off)
  /topp <x>       nucleus cutoff in (0, 1] (1 = off)
  /penalty <x>    repetition penalty (1 = off)
  /tokens <n>     max new tokens per reply
  /status         show current settings
  /reset          clear the conversation history
  /help           show this help
  /quit           exit
"""

# command -> (state key, parser)
OPTIONS = {
    "/temp": ("temperature", float),
    "/topk": ("top_k", int),
    "/topp": ("top_p", float),
    "/penalty": ("repetition_penalty", float),
    "/tokens": ("max_new_tokens", int),
}


def parse_args():
    p = argparse.ArgumentParser(description="Chat with a TinyGPT SFT checkpoint.")
    p.add_argument("--ckpt", default="checkpoints/tinygpt-chat.pt")
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    p.add_argument("--tokens", type=int, default=256, help="max new tokens per reply")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--kv-cache", action="store_true",
                   help="reuse attention keys/values while replying")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    return p.parse_args()


def _show_state(state, system):
    top_k = state["top_k"] if state["top_k"] is not None else "off"
    top_p = state["top_p"] if state["top_p"] is not None else "off"
    print(
        f"system={system!r}\n"
        f"temperature={state['temperature']} top_k={top_k} top_p={top_p} "
        f"repetition_penalty={state['repetition_penalty']} "
        f"max_new_tokens={state['max_new_tokens']}"
    )


def _set_option(state, cmd, raw):
    name, cast = OPTIONS[cmd]
    if raw is None:
        raise ValueError(f"{cmd} needs a value")
    try:
        value = cast(raw)
    except ValueError:
        raise ValueError(f"{cmd} expects a number, got {raw!r}") from None

    if name == "temperature" and value < 0:
        raise ValueError("temperature must be >= 0")
    if name == "max_new_tokens" and value < 1:
        raise ValueError("/tokens must be >= 1")
    if name == "top_k":
        if value < 0:
            raise ValueError("/topk must be >= 0 (0 = off)")
        value = None if value == 0 else value
    if name == "top_p" and not 0.0 < value <= 1.0:
        raise ValueError("/topp must be in (0, 1]")
    if name == "repetition_penalty" and value <= 0:
        raise ValueError("/penalty must be > 0")

    state[name] = value
    print(f"{name} = {value if value is not None else 'off'}")


def reply(model, tok, messages, args, state, system=None):
    """Generate one assistant reply, streaming tokens as they arrive."""
    ids = build_chat_prompt(
        messages,
        tok,
        model.block_size,
        add_generation_prompt=True,
        system=system,
    )
    idx = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)

    pieces = []
    for out in model.generate_stream(
        idx,
        state["max_new_tokens"],
        temperature=state["temperature"],
        top_k=state["top_k"],
        top_p=state["top_p"],
        repetition_penalty=state["repetition_penalty"],
        use_cache=args.kv_cache,
    ):
        token_id = out[0, -1].item()
        if tok.is_special_id(token_id):
            break
        piece = tok.decode([token_id])
        pieces.append(piece)
        print(piece, end="", flush=True)

    return "".join(pieces)


def main():
    args = parse_args()
    if args.tokens < 1:
        raise SystemExit("error: --tokens must be >= 1")
    if args.temperature < 0:
        raise SystemExit("error: --temperature must be >= 0")
    if args.top_k < 0:
        raise SystemExit("error: --top-k must be >= 0 (0 = off)")
    if args.top_p is not None and not 0.0 < args.top_p <= 1.0:
        raise SystemExit("error: --top-p must be in (0, 1]")
    if args.repetition_penalty <= 0:
        raise SystemExit("error: --repetition-penalty must be > 0")

    try:
        model, tok = load_model(args.ckpt, get_device(args.device))
    except (OSError, ValueError, KeyError) as e:
        raise SystemExit(f"error: could not load {args.ckpt}: {e}")

    missing = [t for t in CHAT_SPECIAL_TOKENS if t not in tok.special_tokens]
    if missing:
        shown = ", ".join(missing)
        raise SystemExit(
            f"error: {args.ckpt} is a completion checkpoint, not an SFT chat "
            f"checkpoint (missing {shown})"
        )
    if args.seed is not None:
        set_seed(args.seed)

    state = {
        "max_new_tokens": args.tokens,
        "temperature": args.temperature,
        "top_k": args.top_k or None,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    system = args.system
    messages = []

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("TinyGPT chat")
    print(
        f"checkpoint: {args.ckpt} | {n_params:.2f}M params | "
        f"context {model.block_size} | /help for commands"
    )
    _show_state(state, system)

    while True:
        try:
            line = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not line.strip():
            continue

        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            raw = parts[1] if len(parts) > 1 else None
            if cmd in ("/quit", "/exit"):
                print("bye")
                return
            if cmd in ("/help", "/?"):
                print(HELP)
                continue
            if cmd == "/status":
                _show_state(state, system)
                continue
            if cmd == "/reset":
                messages = []
                print("conversation cleared")
                continue
            if cmd == "/system":
                if not raw or not raw.strip():
                    print("error: /system needs a prompt")
                    continue
                system = raw.strip()
                messages = []
                print(f"system = {system!r}; conversation cleared")
                continue
            if cmd in OPTIONS:
                try:
                    _set_option(state, cmd, raw)
                except ValueError as e:
                    print(f"error: {e}")
                continue
            print(f"unknown command {cmd!r}; /help for a list")
            continue

        messages.append({"role": "user", "content": line.strip()})
        print("\nAssistant: ", end="")
        try:
            answer = reply(model, tok, messages, args, state, system=system)
            print()
        except (KeyboardInterrupt, ValueError) as e:
            print(f"\n[stopped: {e}]")
            messages.pop()
            continue

        if answer.strip():
            messages.append({"role": "assistant", "content": answer.strip()})
        else:
            print("[empty reply]")
            messages.pop()


if __name__ == "__main__":
    main()
