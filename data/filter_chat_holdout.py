"""Filter a held-out chat set to characters seen in the training text."""
import argparse
import json
from pathlib import Path

from training.sft import conversation_text, load_conversations


def filter_known(train, validation):
    """Return retained conversations and aggregate coverage counts."""
    known = set(conversation_text(train))
    kept = [conv for conv in validation if set(conversation_text([conv])) <= known]
    return kept, {"train": len(train), "validation": len(validation),
                  "retained": len(kept), "excluded": len(validation) - len(kept)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--output", required=True, help="new JSONL file; never overwritten")
    args = p.parse_args()
    train = load_conversations(args.train)
    validation = load_conversations(args.validation)
    kept, counts = filter_known(train, validation)
    if not kept:
        p.error("no validation conversations remain")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        for conversation in kept:
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
