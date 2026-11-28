"""Tests for deterministic full-conversation chat evaluation."""
import math
import json
import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

import torch

from chat.template import CHAT_SPECIAL_TOKENS
from checkpoint import save_checkpoint
from inference.evaluate_chat import evaluate_conversations
from model.model import TinyGPT
from tokenizer import ByteTokenizer


class UniformModel(torch.nn.Module):
    def __init__(self, vocab_size, block_size=3):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size
        self.block_size = block_size

    def forward(self, x):
        return torch.zeros((*x.shape, self.vocab_size), device=x.device), None


def test_full_evaluation_counts_each_assistant_target():
    tok = ByteTokenizer(CHAT_SPECIAL_TOKENS)
    model = UniformModel(tok.vocab_size)
    conversations = [
        {"messages": [{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": "é😀"}]},
        {"messages": [{"role": "user", "content": "hello"},
                      {"role": "assistant", "content": "a"}]},
        {"messages": [{"role": "user", "content": "no answer"}]},
    ]
    before = torch.random.get_rng_state().clone()
    result = evaluate_conversations(model, tok, conversations, "cpu")
    assert result["assistant_target_tokens"] == len("é😀".encode()) + 1 + 2
    assert result["scored_conversations"] == 2
    assert math.isclose(result["loss_nats_per_token"], math.log(tok.vocab_size), rel_tol=1e-6)
    assert math.isclose(result["perplexity"], tok.vocab_size, rel_tol=1e-6)
    assert model.training
    assert torch.equal(before, torch.random.get_rng_state())
    assert result == evaluate_conversations(model, tok, conversations, "cpu")


def test_full_evaluation_restores_mode_on_error():
    tok = ByteTokenizer(CHAT_SPECIAL_TOKENS)
    model = UniformModel(tok.vocab_size)
    model.eval()
    with patch.object(model, "forward", side_effect=RuntimeError("failure")):
        try:
            evaluate_conversations(model, tok, [{"messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"}]}], "cpu")
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected model failure")
    assert not model.training


def test_full_evaluation_rejects_no_targets_or_missing_chat_tokens():
    convs = [{"messages": [{"role": "user", "content": "q"}]}]
    for tok in (ByteTokenizer(), ByteTokenizer(CHAT_SPECIAL_TOKENS)):
        try:
            evaluate_conversations(UniformModel(tok.vocab_size), tok, convs, "cpu")
        except ValueError:
            pass
        else:
            raise AssertionError("expected invalid evaluation")


def test_full_evaluation_cli_writes_report():
    tok = ByteTokenizer(CHAT_SPECIAL_TOKENS)
    cfg = dict(vocab_size=tok.vocab_size, block_size=8, n_embd=8, n_head=2, n_layer=1)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "model.pt")
        data = os.path.join(tmp, "val.jsonl")
        report = os.path.join(tmp, "report.json")
        save_checkpoint(ckpt, TinyGPT(**cfg), None, cfg, tok.chars, 0, [], 1.0,
                        tokenizer=tok.to_dict())
        with open(data, "w", encoding="utf-8") as f:
            f.write(json.dumps({"messages": [{"role": "user", "content": "Hi"},
                                             {"role": "assistant", "content": "é"}]}) + "\n")
        command = [sys.executable, "-m", "inference.evaluate_chat", "--ckpt", ckpt,
                   "--data", data, "--device", "cpu", "--output", report]
        subprocess.run(command, check=True, capture_output=True)
        with open(report, encoding="utf-8") as f:
            result = json.load(f)
        assert result["metrics"]["assistant_target_tokens"] == 3
        assert subprocess.run(command, capture_output=True).returncode != 0
