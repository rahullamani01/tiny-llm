"""Tests for the BPE tokenizer and tokenizer-aware checkpoints."""
import os
import random
import tempfile

import torch

from checkpoint import load_checkpoint, save_checkpoint
from inference.generate import load_model
from model.model import TinyGPT
from tokenizer import (
    BPETokenizer,
    CharTokenizer,
    load_tokenizer,
    save_tokenizer,
    tokenizer_from_state,
)

TEXT = "the quick brown fox jumps over the lazy dog. " * 30


def _trained(vocab_size=80, min_frequency=2):
    return BPETokenizer.train(
        TEXT,
        vocab_size=vocab_size,
        max_chars=len(TEXT),
        min_frequency=min_frequency,
    )


def test_bpe_roundtrip():
    tok = _trained()
    assert tok.vocab_size > len(tok.chars)
    for s in (TEXT, "the quick brown fox", "lazy dog", ""):
        ids = tok.encode(s)
        assert tok.decode(ids) == s
        assert ids == tok._encode_naive(s)


def test_bpe_heap_matches_naive_on_random_text():
    tok = _trained()
    rng = random.Random(0)
    alphabet = "".join(tok.chars)
    for _ in range(20):
        s = "".join(rng.choice(alphabet) for _ in range(40))
        ids = tok.encode(s)
        assert ids == tok._encode_naive(s)
        assert tok.decode(ids) == s


def test_bpe_vocab_size_and_determinism():
    tok = _trained(vocab_size=60)
    assert tok.vocab_size <= 60
    assert tok.vocab_size == len(tok.chars) + len(tok.merges)

    again = _trained(vocab_size=60)
    assert again.to_dict() == tok.to_dict()


def test_bpe_min_frequency_stops_merges():
    tok = BPETokenizer.train("ab", vocab_size=10, max_chars=2, min_frequency=5)
    assert tok.vocab_size == 2
    assert tok.merges == []


def test_bpe_unknown_chars():
    tok = BPETokenizer.train("abc", vocab_size=10, max_chars=3)
    assert tok.missing_chars("cab") == []
    assert tok.missing_chars("cabx") == ["x"]


def test_tokenizer_state_roundtrip():
    char_tok = CharTokenizer("abcabc")
    char_again = tokenizer_from_state(char_tok.to_dict())
    assert isinstance(char_again, CharTokenizer)
    assert char_again.chars == char_tok.chars

    bpe_tok = _trained()
    bpe_again = tokenizer_from_state(bpe_tok.to_dict())
    assert isinstance(bpe_again, BPETokenizer)
    assert bpe_again.to_dict() == bpe_tok.to_dict()
    sample = "the lazy dog"
    assert bpe_again.encode(sample) == bpe_tok.encode(sample)


def test_save_load_tokenizer():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nested", "vocab.json")
        tok = _trained()
        save_tokenizer(tok, path)
        again = load_tokenizer(path)
        assert again.to_dict() == tok.to_dict()
        assert again.encode(TEXT) == tok.encode(TEXT)


def test_bpe_checkpoint_roundtrip_and_generation():
    tok = BPETokenizer.train("abcabcabcabc", vocab_size=8, max_chars=12)
    cfg = dict(
        vocab_size=tok.vocab_size,
        block_size=8,
        n_embd=16,
        n_head=2,
        n_layer=1,
        dropout=0.0,
        tie_weights=False,
    )
    model = TinyGPT(**cfg)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bpe.pt")
        save_checkpoint(
            path,
            model,
            None,
            cfg,
            tok.chars,
            0,
            [],
            float("inf"),
            tokenizer=tok.to_dict(),
        )
        ckpt = load_checkpoint(path, "cpu")
        assert ckpt["tokenizer"]["type"] == "bpe"

        model2, tok2 = load_model(path, "cpu")
        assert isinstance(tok2, BPETokenizer)
        assert tok2.to_dict() == tok.to_dict()
        ids = tok2.encode("abc")
        out = model2.generate(torch.tensor([ids]), 5, temperature=0.0)
        assert out.shape == (1, len(ids) + 5)
        assert tok2.decode(out[0].tolist()).startswith("abc")


def test_legacy_checkpoint_uses_char_tokenizer():
    tok = CharTokenizer("abc")
    cfg = dict(
        vocab_size=tok.vocab_size,
        block_size=8,
        n_embd=16,
        n_head=2,
        n_layer=1,
        dropout=0.0,
        tie_weights=False,
    )
    model = TinyGPT(**cfg)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "legacy.pt")
        torch.save(
            {"model": model.state_dict(), "config": cfg, "chars": tok.chars},
            path,
        )
        _model, tok2 = load_model(path, "cpu")
        assert isinstance(tok2, CharTokenizer)
        assert tok2.chars == tok.chars


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok:", name)
