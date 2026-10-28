"""Train a BPE vocabulary and save it as JSON.

    python -m tokenizer.train_bpe --data data/training.txt \
        --vocab-size 512 --out checkpoints/bpe.json
"""
import argparse

from tokenizer import BPETokenizer, save_tokenizer


def main():
    p = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    p.add_argument("--data", default="data/training.txt")
    p.add_argument("--out", default="checkpoints/bpe.json")
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--max-chars", type=int, default=100_000,
                   help="text prefix used to learn merges")
    p.add_argument("--min-frequency", type=int, default=2)
    a = p.parse_args()

    try:
        text = open(a.data, encoding="utf-8").read()
    except FileNotFoundError:
        raise SystemExit(f"error: data file not found: {a.data}\nrun: python data/download.py")

    try:
        tok = BPETokenizer.train(
            text,
            a.vocab_size,
            max_chars=a.max_chars,
            min_frequency=a.min_frequency,
        )
    except ValueError as e:
        raise SystemExit(f"error: {e}")

    save_tokenizer(tok, a.out)
    print(
        f"vocab: {tok.vocab_size} ({len(tok.chars)} chars + "
        f"{len(tok.merges)} merges) | saved {a.out}"
    )
    sample = text[:200]
    assert tok.decode(tok.encode(sample)) == sample
    print(repr(tok.decode(tok.encode(sample))))


if __name__ == "__main__":
    main()
