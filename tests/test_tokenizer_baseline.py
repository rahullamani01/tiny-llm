"""Tests for the tokenizer comparison experiment."""
from types import SimpleNamespace
import math

import torch

from experiments.tokenizer_baseline import evaluate_metrics, run_one, split_text
from tokenizer import CharTokenizer


def test_text_is_split_before_tokenization():
    train, val = split_text("abcdefghij", 0.2)
    assert train == "abcdefgh"
    assert val == "ij"
    for fraction in (0, 1, -0.1):
        try:
            split_text("abc", fraction)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid split accepted")


def test_baselines_run_on_same_text():
    args = SimpleNamespace(vocab_size=8, bpe_max_chars=100, seed=3,
                           block_size=4, n_embd=8, n_head=2, n_layer=1,
                           lr=0.001, steps=2, batch_size=2, eval_iters=2,
                           prompts=["ab"], sample_tokens=2, eval_interval=1)
    text = "abcd" * 30
    for kind in ("char", "bpe"):
        result = run_one(kind, text, text, args, torch.device("cpu"))
        assert result["tokenizer"] == kind
        assert result["val_loss_nats_per_token"] > 0
        assert len(result["history"]) == 2
        assert result["samples_greedy"]["ab"].startswith("ab")


def test_validation_characters_do_not_leak_into_vocabulary():
    args = SimpleNamespace(vocab_size=8, bpe_max_chars=100)
    for kind in ("char", "bpe"):
        try:
            run_one(kind, "abcd" * 10, "abcdz", args, torch.device("cpu"))
        except ValueError as exc:
            assert "absent from training" in str(exc)
        else:
            raise AssertionError("unseen validation character was accepted")


def test_bits_per_byte_counts_utf8_target_bytes():
    class UniformModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((*x.shape, 2)), None

    tok = CharTokenizer("aé")
    model = UniformModel()
    data = torch.tensor(tok.encode("é" * 20))
    before = torch.random.get_rng_state().clone()
    metrics = evaluate_metrics(model, data, tok, 4, 2, 3, "cpu", 42)
    assert math.isclose(metrics["bits_per_byte"], 0.5, abs_tol=1e-6)
    assert metrics["target_bytes"] == 2 * metrics["target_tokens"]
    assert model.training
    assert torch.equal(before, torch.random.get_rng_state())
