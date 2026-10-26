"""Interactive, streaming text-generation REPL.

Type a prompt and the model keeps writing from the running context, one
character at a time. Slash commands change the sampling settings on the fly.

    python -m inference.repl
"""
import argparse

from inference.generate import load_model, stream_text
from utils import set_seed

HELP = """commands:
  /temp <x>      temperature (0 = greedy, higher = wilder)
  /topk <n>      keep the n most likely tokens (0 = off)
  /topp <x>      nucleus cutoff in (0, 1] (1 = off)
  /penalty <x>   repetition penalty (1 = off, higher = less repetition)
  /tokens <n>    max new tokens per turn
  /status        show the current settings
  /reset         forget the running context
  /help          show this help
  /quit          exit
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
    p = argparse.ArgumentParser(description="Interactive streaming generation.")
    p.add_argument("--ckpt", default="checkpoints/tinygpt.pt")
    p.add_argument("--prompt", default="\n", help="initial context")
    p.add_argument("--tokens", type=int, default=300, help="max new tokens per turn")
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=40, help="0 disables top-k")
    p.add_argument("--top-p", type=float, default=None, help="nucleus cutoff (off by default)")
    p.add_argument("--repetition-penalty", type=float, default=1.0, help="1.0 = off")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def _show_state(state):
    top_k = state["top_k"] if state["top_k"] is not None else "off"
    top_p = state["top_p"] if state["top_p"] is not None else "off"
    print(
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
    shown = value if value is not None else "off"
    print(f"{name} = {shown}")


def main():
    args = parse_args()
    model, tok = load_model(args.ckpt)
    if args.seed is not None:
        set_seed(args.seed)

    state = {
        "max_new_tokens": args.tokens,
        "temperature": args.temperature,
        "top_k": args.top_k or None,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    context = args.prompt

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("tiny-llm interactive generation")
    print(f"checkpoint: {args.ckpt} | {n_params:.2f}M params | /help for commands")
    _show_state(state)

    while True:
        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

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
                _show_state(state)
                continue
            if cmd == "/reset":
                context = ""
                print("context cleared")
                continue
            if cmd in OPTIONS:
                try:
                    _set_option(state, cmd, raw)
                except ValueError as e:
                    print(f"error: {e}")
                continue
            print(f"unknown command {cmd!r}; /help for a list")
            continue

        if not line:
            line = "\n"
        candidate = context + line
        if not candidate.endswith("\n"):
            candidate += "\n"

        generated = []
        print()
        try:
            for chunk in stream_text(model, tok, candidate, **state):
                generated.append(chunk)
                print(chunk, end="", flush=True)
        except KeyboardInterrupt:
            print("\n[stopped]")
        except ValueError as e:
            print(f"error: {e}")
            continue
        print()

        # Keep a rolling context so the next turn continues the same text.
        # Char tokenizer: one character == one token, so block_size chars fit.
        context = (candidate + "".join(generated))[-model.block_size :]


if __name__ == "__main__":
    main()
