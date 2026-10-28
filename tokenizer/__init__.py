"""Tokenizer implementations and JSON state helpers."""
import json
import os

from tokenizer.bpe import BPETokenizer
from tokenizer.tokenizer import CharTokenizer

__all__ = [
    "BPETokenizer",
    "CharTokenizer",
    "load_tokenizer",
    "save_tokenizer",
    "tokenizer_from_state",
]


def tokenizer_from_state(state):
    kind = state.get("type")
    if kind == "char":
        return CharTokenizer.from_dict(state)
    if kind == "bpe":
        return BPETokenizer.from_dict(state)
    raise ValueError(f"unknown tokenizer type {kind!r}")


def save_tokenizer(tok, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tok.to_dict(), f, indent=2)


def load_tokenizer(path):
    with open(path, encoding="utf-8") as f:
        return tokenizer_from_state(json.load(f))
