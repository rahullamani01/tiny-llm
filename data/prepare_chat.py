"""Convert public instruction datasets into chat JSONL for ``training.sft``.

    # download Alpaca (52k instructions), keep 8k, write chat + plain text
    python -m data.prepare_chat --source alpaca --limit 8000 \
        --out data/cache/chat.jsonl --text-out data/cache/text.txt

    # Dolly 15k works too (instruction/context/response)
    python -m data.prepare_chat --source dolly --out data/cache/dolly.jsonl

    # convert a local file in either shape
    python -m data.prepare_chat --source input --input my_data.jsonl --out ...

Both Alpaca (``instruction``/``input``/``output``) and Dolly
(``instruction``/``context``/``response``) records are accepted. Output is the
conversation format expected by SFT::

    {"system": "...", "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}

``--format text`` (or ``--text-out``) writes plain text for Base pretraining.
The text contains no chat special tokens, so the SFT tokenizer can add them
later without collisions.
"""
import argparse
import json
import os
import random
import shutil
import ssl
import tempfile
import urllib.request

DEFAULT_SYSTEM = "You are a helpful, concise assistant."
DEFAULT_MAX_CHARS = 2000
DEFAULT_DOWNLOAD_DIR = "data/cache"

# name -> (filename, url)
SOURCES = {
    "alpaca": (
        "alpaca_data.json",
        "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
        "main/alpaca_data.json",
    ),
    "dolly": (
        "databricks-dolly-15k.jsonl",
        "https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
        "resolve/main/databricks-dolly-15k.jsonl",
    ),
}


def clean_text(value):
    """Normalize line endings and trim whitespace without destroying layout."""
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


def record_to_conversation(record, system=DEFAULT_SYSTEM, max_chars=DEFAULT_MAX_CHARS):
    """Convert one Alpaca/Dolly record, or return ``None`` when unusable.

    Empty instructions/answers and examples longer than ``max_chars`` are
    skipped so a single huge sample cannot dominate the SFT windows.
    """
    if not isinstance(record, dict):
        return None

    instruction = clean_text(record.get("instruction") or record.get("prompt"))
    answer = clean_text(
        record.get("response") or record.get("output") or record.get("completion")
    )
    # Dolly uses "context"; Alpaca uses "input" for the optional passage.
    extra = clean_text(record.get("context"))
    extra_label = "Context"
    if not extra:
        extra = clean_text(record.get("input"))
        extra_label = "Input"

    if not instruction or not answer:
        return None

    user = instruction
    if extra:
        user = f"{instruction}\n\n{extra_label}:\n{extra}"
    if max_chars and len(user) + len(answer) > max_chars:
        return None

    return {
        "system": system,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
    }


def convert_records(records, system=DEFAULT_SYSTEM, max_chars=DEFAULT_MAX_CHARS):
    """Convert an iterable of records, dropping invalid and duplicate examples."""
    conversations = []
    seen = set()
    skipped = 0
    for record in records:
        conversation = record_to_conversation(
            record, system=system, max_chars=max_chars
        )
        if conversation is None:
            skipped += 1
            continue
        key = (
            conversation["messages"][0]["content"],
            conversation["messages"][1]["content"],
        )
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        conversations.append(conversation)
    return conversations, skipped


def conversation_to_text(conversation):
    """Plain training text for the Base model (no chat special tokens)."""
    user, assistant = conversation["messages"]
    return f"{user['content']}\n\n{assistant['content']}"


def load_records(path):
    """Read a JSON array (Alpaca) or JSONL file (Dolly) into a list of records."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        raise ValueError(f"data file not found: {path}") from None

    if not text:
        raise ValueError(f"{path}: file is empty")

    if text.startswith("["):
        try:
            records = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: invalid JSON: {e}") from None
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a JSON array of records")
        return records

    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from None
        records.append(record)
    return records


def download_source(source, download_dir=DEFAULT_DOWNLOAD_DIR, timeout=120):
    """Download and cache the named dataset, returning the local path."""
    filename, url = SOURCES[source]
    path = os.path.join(download_dir, filename)
    if os.path.exists(path):
        print(f"using cached {path}")
        return path

    os.makedirs(download_dir, exist_ok=True)
    print(f"downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "tiny-llm/1.0"})
    # Some Python installs ship without a usable system CA bundle.
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    fd, tmp_path = tempfile.mkstemp(dir=download_dir, suffix=".part")
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=context
        ) as response:
            with os.fdopen(fd, "wb") as out:
                shutil.copyfileobj(response, out)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print(f"saved {path}")
    return path


def write_chat_jsonl(conversations, path):
    """Write one conversation per line, creating parent directories."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for conversation in conversations:
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")


def write_text(conversations, path):
    """Write plain text examples separated by blank lines."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for conversation in conversations:
            f.write(conversation_to_text(conversation) + "\n\n")


def prepare(
    source,
    input_path=None,
    download_dir=DEFAULT_DOWNLOAD_DIR,
    system=DEFAULT_SYSTEM,
    max_chars=DEFAULT_MAX_CHARS,
    limit=0,
    shuffle=False,
    seed=0,
):
    """Load, convert and (optionally) subset a dataset. Returns conversations."""
    if limit < 0:
        raise ValueError("--limit must be >= 0")
    if max_chars < 0:
        raise ValueError("--max-chars must be >= 0")
    if source == "input":
        if not input_path:
            raise ValueError("--source input requires --input")
        path = input_path
    else:
        path = input_path or download_source(source, download_dir)

    records = load_records(path)
    conversations, skipped = convert_records(
        records, system=system, max_chars=max_chars
    )
    if shuffle:
        random.Random(seed).shuffle(conversations)
    if limit:
        conversations = conversations[:limit]
    if not conversations:
        raise ValueError("no usable conversations found; check the input format")
    print(
        f"{len(records)} records -> {len(conversations)} conversations "
        f"({skipped} skipped)"
    )
    return conversations


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert instruction data to chat JSONL for training.sft."
    )
    p.add_argument(
        "--source",
        choices=sorted(SOURCES) + ["input"],
        default="alpaca",
        help="dataset to download, or 'input' for --input",
    )
    p.add_argument("--input", default=None, help="local JSON/JSONL file")
    p.add_argument("--download-dir", default=DEFAULT_DOWNLOAD_DIR)
    p.add_argument("--out", default=None, help="chat JSONL output path")
    p.add_argument("--val-out", default=None, help="hold out prompt groups to this JSONL file")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument(
        "--text-out",
        default=None,
        help="also write plain text for Base pretraining",
    )
    p.add_argument(
        "--format",
        choices=["chat", "text"],
        default="chat",
        help="what --out contains when --text-out is not used",
    )
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                   help="drop examples longer than this (0 = keep all)")
    p.add_argument("--limit", type=int, default=0,
                   help="keep at most N conversations (0 = all)")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        conversations = prepare(
            args.source,
            input_path=args.input,
            download_dir=args.download_dir,
            system=args.system,
            max_chars=args.max_chars,
            limit=args.limit,
            shuffle=args.shuffle,
            seed=args.seed,
        )
    except (OSError, ValueError) as e:
        raise SystemExit(f"error: {e}") from None

    base = args.out or os.path.join(
        args.download_dir,
        f"{args.source}_{args.format}." + ("jsonl" if args.format == "chat" else "txt"),
    )
    if args.val_out:
        if args.format != "chat":
            raise SystemExit("error: --val-out requires --format chat")
        paths = [base, args.val_out] + ([args.text_out] if args.text_out else [])
        if len({os.path.abspath(path) for path in paths}) != len(paths):
            raise SystemExit("error: output paths must be distinct")
        from training.sft import split_conversations
        try:
            conversations, validation = split_conversations(
                conversations, args.val_fraction, args.seed
            )
        except ValueError as e:
            raise SystemExit(f"error: {e}") from None
        write_chat_jsonl(validation, args.val_out)
        print(f"wrote {len(validation)} held-out conversations to {args.val_out}")
    if args.format == "text":
        write_text(conversations, base)
        print(f"wrote {len(conversations)} text examples to {base}")
    else:
        write_chat_jsonl(conversations, base)
        print(f"wrote {len(conversations)} conversations to {base}")

    if args.text_out:
        write_text(conversations, args.text_out)
        print(f"wrote {len(conversations)} text examples to {args.text_out}")


if __name__ == "__main__":
    main()
