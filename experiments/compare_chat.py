"""Compare a base and SFT checkpoint on a fixed set of prompts."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from chat.template import CHAT_SPECIAL_TOKENS, build_chat_prompt
from inference.chat import DEFAULT_SYSTEM
from inference.generate import load_model
from tokenizer import incremental_decoder
from utils import get_device


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete(model, tok, prompt, max_tokens, chat=False, system=DEFAULT_SYSTEM):
    """Return greedy continuation and stopping reason, excluding the prompt."""
    if not prompt or max_tokens < 1:
        raise ValueError("prompt must be nonempty and max_tokens must be positive")
    if chat:
        missing = [token for token in CHAT_SPECIAL_TOKENS if token not in tok.special_tokens]
        if missing:
            raise ValueError("chat checkpoint is missing chat special tokens")
        ids = build_chat_prompt([{"role": "user", "content": prompt}], tok,
                                model.block_size, system=system)
    else:
        ids = tok.encode(prompt + "\n\n")[-model.block_size:]
    idx = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    pieces = []
    decoder = incremental_decoder(tok)
    stop = "token_limit"
    model.eval()
    for out in model.generate_stream(idx, max_tokens, temperature=0):
        token = out[0, -1].item()
        if tok.is_special_id(token):
            stop = "special_token"
            break
        pieces.append(decoder.push(token))
    pieces.append(decoder.finish())
    return {"text": "".join(pieces), "stop": stop}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--chat", required=True)
    p.add_argument("--prompts", default="experiments/chat_prompts.json")
    p.add_argument("--output", required=True, help="new JSON file, never overwritten")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = p.parse_args()
    if args.tokens < 1:
        p.error("--tokens must be positive")
    if Path(args.output).exists():
        p.error("output already exists")
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts:
        p.error("prompts must be a nonempty JSON array")
    device = get_device(args.device)
    base, base_tok = load_model(args.base, device)
    chat, chat_tok = load_model(args.chat, device)
    results = []
    for item in prompts:
        prompt = item["prompt"]
        results.append({
            "category": item["category"], "prompt": prompt,
            "base": complete(base, base_tok, prompt, args.tokens),
            "chat": complete(chat, chat_tok, prompt, args.tokens, chat=True),
        })
    report = {
        "base_sha256": file_hash(args.base), "chat_sha256": file_hash(args.chat),
        "prompts_sha256": file_hash(args.prompts), "device": str(device),
        "torch_version": torch.__version__, "max_new_tokens": args.tokens,
        "decoding": "greedy", "chat_system": DEFAULT_SYSTEM, "results": results,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
