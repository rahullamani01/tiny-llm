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
    "with_special_tokens",
]


def tokenizer_from_state(state):
    kind = state.get("type")
    if kind == "char":
        return CharTokenizer.from_dict(state)
    if kind == "bpe":
        return BPETokenizer.from_dict(state)
    raise ValueError(f"unknown tokenizer type {kind!r}")


def with_special_tokens(tok, special_tokens):
    """Return a copy of ``tok`` with extra special tokens appended.

    Existing ids never move, so a model trained with the old vocabulary can keep
    its old embedding rows and only needs new rows for the special tokens.
    """
    wanted = list(tok.special_tokens)
    for token in special_tokens:
        if token not in wanted:
            wanted.append(token)
    if type(tok) is CharTokenizer:
        return CharTokenizer.from_chars(tok.chars, wanted)
    if type(tok) is BPETokenizer:
        return BPETokenizer(tok.chars, tok.merges, wanted)
    raise TypeError(f"unsupported tokenizer type {type(tok).__name__}")


def save_tokenizer(tok, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tok.to_dict(), f, indent=2)


def load_tokenizer(path):
    with open(path, encoding="utf-8") as f:
        return tokenizer_from_state(json.load(f))
