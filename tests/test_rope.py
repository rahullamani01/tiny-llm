"""Tests for rotary position embeddings (RoPE). Run: python -m tests.test_rope"""
import os
import tempfile

import torch

from checkpoint import save_checkpoint
from inference.generate import load_model
from model.model import TinyGPT
from model.rope import RotaryEmbedding
from tokenizer import CharTokenizer


def _model(pos_embedding="rope", block_size=16, n_layer=2, n_head=4, n_embd=32):
    torch.manual_seed(0)
    return TinyGPT(
        vocab_size=20,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        pos_embedding=pos_embedding,
    ).eval()


def _tokens(batch, length, vocab_size=20, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (batch, length), generator=generator)


def test_rope_preserves_norms():
    rope = RotaryEmbedding(8, 64)
    x = torch.randn(2, 5, 8)
    y = rope(x, pos_offset=3)
    torch.testing.assert_close(
        y.norm(dim=-1), x.norm(dim=-1), atol=1e-6, rtol=1e-6
    )


def test_rope_position_zero_is_identity():
    rope = RotaryEmbedding(8, 64)
    x = torch.randn(2, 1, 8)
    torch.testing.assert_close(rope(x, pos_offset=0), x, atol=1e-6, rtol=1e-6)


def test_rope_scores_depend_only_on_relative_position():
    rope = RotaryEmbedding(8, 64)
    q = torch.randn(1, 1, 8)
    k = torch.randn(1, 1, 8)

    for query_pos, key_pos in ((1, 0), (7, 3), (20, 11)):
        close = rope(q, query_pos) @ rope(k, key_pos).transpose(-1, -2)
        shifted = rope(q, query_pos + 9) @ rope(k, key_pos + 9).transpose(-1, -2)
        torch.testing.assert_close(close, shifted, atol=1e-5, rtol=1e-5)


def test_rope_scores_change_with_distance():
    rope = RotaryEmbedding(8, 64)
    q = torch.randn(1, 1, 8)
    k = torch.randn(1, 1, 8)

    near = rope(q, 10) @ rope(k, 8).transpose(-1, -2)
    far = rope(q, 10) @ rope(k, 2).transpose(-1, -2)
    assert not torch.allclose(near, far, atol=1e-4)


def test_rope_model_has_no_position_table():
    model = _model("rope")
    assert model.pos_embedding == "rope"
    assert model.embed.position_embedding is None
    names = [name for name, _ in model.named_parameters()]
    assert not any("position_embedding" in name for name in names)
    assert "embed.token_embedding.weight" in names


def test_learned_model_keeps_position_table():
    model = _model("learned")
    assert model.pos_embedding == "learned"
    assert model.embed.position_embedding is not None
    assert model.embed.position_embedding.weight.shape == (16, 32)


def test_rope_cached_logits_match_uncached():
    model = _model("rope")
    x = _tokens(batch=2, length=12, seed=1)

    with torch.no_grad():
        full, _ = model(x)
        pieces = []
        logits, _, past = model(x[:, :7], use_cache=True)
        pieces.append(logits)
        for t in range(7, 12):
            logits, _, past = model(
                x[:, t:t + 1], past_key_values=past, use_cache=True
            )
            pieces.append(logits)
        incremental = torch.cat(pieces, dim=1)

    torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-4)


def test_rope_cached_greedy_matches_uncached_across_window():
    model = _model("rope", block_size=8)
    prompt = _tokens(batch=2, length=3, seed=2)

    uncached = model.generate(prompt, 20, temperature=0.0, use_cache=False)
    cached = model.generate(prompt, 20, temperature=0.0, use_cache=True)

    assert cached.shape == (2, 23)
    assert torch.equal(cached, uncached)


def test_rope_training_step():
    torch.manual_seed(0)
    model = TinyGPT(
        vocab_size=20,
        block_size=16,
        n_embd=32,
        n_head=4,
        n_layer=2,
        pos_embedding="rope",
    )
    x = _tokens(batch=2, length=8, seed=3)
    _logits, loss = model(x, x)
    loss.backward()

    head = model.blocks[0].sa.heads[0]
    assert head.query.weight.grad is not None
    assert head.key.weight.grad is not None
    assert head.value.weight.grad is not None


def test_rope_checkpoint_roundtrip():
    tok = CharTokenizer("abc")
    cfg = dict(
        vocab_size=tok.vocab_size,
        block_size=8,
        n_embd=16,
        n_head=2,
        n_layer=1,
        dropout=0.0,
        tie_weights=False,
        pos_embedding="rope",
    )
    model = TinyGPT(**cfg)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rope.pt")
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
        loaded, tok2 = load_model(path, "cpu")

    assert tok2.chars == tok.chars
    assert loaded.pos_embedding == "rope"
    assert loaded.embed.position_embedding is None

    x = torch.tensor([[0, 1, 2]])
    with torch.no_grad():
        torch.testing.assert_close(model(x)[0], loaded(x)[0], atol=1e-6, rtol=1e-6)


def test_legacy_config_defaults_to_learned_positions():
    model = TinyGPT(vocab_size=20, block_size=8, n_embd=16, n_head=2, n_layer=1)
    assert model.pos_embedding == "learned"
    assert model.embed.position_embedding is not None


def test_resume_rejects_changed_position_scheme():
    from training.train import _checkpoint_matches

    base = dict(
        vocab_size=20,
        block_size=8,
        n_embd=16,
        n_head=2,
        n_layer=1,
        dropout=0.0,
        tie_weights=False,
    )
    # Legacy configs have no key and mean learned positions.
    _checkpoint_matches(base, {**base, "pos_embedding": "learned"})

    try:
        _checkpoint_matches(
            {**base, "pos_embedding": "rope"},
            {**base, "pos_embedding": "learned"},
        )
    except SystemExit as e:
        assert "pos_embedding" in str(e), e
    else:
        raise AssertionError("expected SystemExit when the position scheme changes")


def test_rope_requires_even_head_size():
    try:
        TinyGPT(
            vocab_size=20,
            block_size=16,
            n_embd=30,
            n_head=2,  # head_size = 15
            n_layer=1,
            pos_embedding="rope",
        )
    except ValueError as e:
        assert "even" in str(e), e
    else:
        raise AssertionError("expected ValueError for an odd head_size")


def test_unknown_position_embedding_is_rejected():
    try:
        TinyGPT(
            vocab_size=20,
            block_size=16,
            n_embd=32,
            n_head=4,
            n_layer=1,
            pos_embedding="alibi",
        )
    except ValueError as e:
        assert "pos_embedding" in str(e), e
    else:
        raise AssertionError("expected ValueError for an unknown pos_embedding")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok:", name)
