"""Sanity checks for training utilities. Run with: python -m tests.test_training"""
import math
import os
import tempfile

import torch

from checkpoint import load_checkpoint, save_checkpoint
from model.model import TinyGPT
from training.dataset import get_batch, split_data
from training.evaluate import evaluate_loss
from training.schedule import cosine_lr, get_lr
from training.train import CFG, run_optimizer_step, validate_config
from utils import get_device, set_seed

MODEL_CFG = dict(
    vocab_size=11,
    block_size=8,
    n_embd=16,
    n_head=2,
    n_layer=1,
    dropout=0.0,
    tie_weights=False,
)


def _model(tied=False):
    torch.manual_seed(0)
    cfg = dict(MODEL_CFG, tie_weights=tied)
    return TinyGPT(**cfg)


def test_cosine_schedule():
    assert math.isclose(cosine_lr(0, 100, 1.0, warmup_steps=4, min_lr=0.1), 0.25)
    assert math.isclose(cosine_lr(3, 100, 1.0, warmup_steps=4, min_lr=0.1), 1.0)
    assert math.isclose(cosine_lr(100, 100, 1.0, warmup_steps=4, min_lr=0.1), 0.1)
    assert get_lr(50, 100, 1.0, "none", warmup_steps=4, min_lr=0.1) == 1.0
    assert 0.0 <= get_lr(99, 100, 1.0, "cosine", warmup_steps=0, min_lr=0.0) < 0.01


def test_schedule_rejects_bad_values():
    bad = [
        dict(max_steps=0),
        dict(warmup_steps=101),
        dict(min_lr=2.0),
        dict(base_lr=-1.0),
    ]
    for override in bad:
        kwargs = dict(step=0, max_steps=100, base_lr=1.0, warmup_steps=0, min_lr=0.0)
        kwargs.update(override)
        try:
            cosine_lr(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {override}")
    try:
        get_lr(0, 100, 1.0, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown schedule")


def test_split_data():
    data = torch.arange(100)
    splits = split_data(data, 0.2)
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 20
    assert torch.equal(torch.cat([splits["train"], splits["val"]]), data)

    for bad in (0.0, 1.0, -0.1, 1.1):
        try:
            split_data(data, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for val_fraction={bad}")


def test_get_batch_shapes_and_shift():
    torch.manual_seed(0)
    data = torch.arange(100)
    x, y = get_batch(data, 8, 4, torch.device("cpu"))
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])
    try:
        get_batch(data[:8], 8, 4, torch.device("cpu"))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for short data")


def test_evaluate_loss():
    model = _model()
    data = torch.randint(0, MODEL_CFG["vocab_size"], (200,))
    loss = evaluate_loss(model, data, 8, batch_size=4, iters=3, device="cpu", seed=0)
    assert math.isfinite(loss)
    assert loss > 0

    try:
        evaluate_loss(model, data[:2], 8, batch_size=4, iters=1, device="cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for short evaluation data")


def test_gradient_accumulation_matches_large_batch():
    torch.manual_seed(0)
    model_a = _model()
    model_b = _model()  # same initialization
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)

    x1 = torch.randint(0, MODEL_CFG["vocab_size"], (3, MODEL_CFG["block_size"]))
    y1 = torch.randint(0, MODEL_CFG["vocab_size"], (3, MODEL_CFG["block_size"]))
    x2 = torch.randint(0, MODEL_CFG["vocab_size"], (3, MODEL_CFG["block_size"]))
    y2 = torch.randint(0, MODEL_CFG["vocab_size"], (3, MODEL_CFG["block_size"]))

    loss_accum = run_optimizer_step(model_a, opt_a, [(x1, y1), (x2, y2)])
    loss_big = run_optimizer_step(
        model_b, opt_b, [(torch.cat([x1, x2]), torch.cat([y1, y2]))]
    )

    assert math.isclose(loss_accum, loss_big, rel_tol=1e-5, abs_tol=1e-6)
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-6, rtol=1e-5)

    try:
        run_optimizer_step(model_a, opt_a, [])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an empty micro-batch list")


def test_validate_config_rejects_bad_batch_options():
    validate_config(dict(CFG))
    for override in ({"grad_accum_steps": 0}, {"weight_decay": -0.1}):
        try:
            validate_config(dict(CFG, **override))
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {override}")


def test_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nested", "model.pt")
        model = _model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        chars = list("abcdefghijk")
        history = [(0, 4.0, 4.1), (5, 3.0, 3.1)]
        save_checkpoint(
            path,
            model,
            opt,
            MODEL_CFG,
            chars,
            step=7,
            history=history,
            best_val_loss=3.1,
            train_config={"data": "x"},
        )

        ckpt = load_checkpoint(path, "cpu")
        assert ckpt["step"] == 7
        assert ckpt["chars"] == chars
        assert ckpt["history"] == history
        assert ckpt["best_val_loss"] == 3.1
        assert ckpt["config"] == MODEL_CFG
        assert ckpt["train_config"] == {"data": "x"}
        assert "optimizer" in ckpt

        model2 = TinyGPT(**ckpt["config"])
        model2.load_state_dict(ckpt["model"])
        for a, b in zip(model.state_dict().values(), model2.state_dict().values()):
            assert torch.equal(a, b)

        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        opt2.load_state_dict(ckpt["optimizer"])


def test_legacy_checkpoint_without_optimizer():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "legacy.pt")
        model = _model()
        torch.save(
            {
                "model": model.state_dict(),
                "config": MODEL_CFG,
                "chars": list("abcdefghijk"),
            },
            path,
        )
        ckpt = load_checkpoint(path, "cpu")
        assert "optimizer" not in ckpt
        assert "step" not in ckpt


def test_tie_weights_share_storage():
    model = _model(tied=True)
    assert model.lm_head.weight.data_ptr() == model.embed.token_embedding.weight.data_ptr()
    x = torch.randint(0, MODEL_CFG["vocab_size"], (2, MODEL_CFG["block_size"]))
    _, loss = model(x, x)
    loss.backward()
    assert model.embed.token_embedding.weight.grad is not None


def test_device_and_seed_helpers():
    assert get_device("cpu").type == "cpu"
    try:
        get_device("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown device")

    set_seed(123)
    a = torch.randn(4)
    set_seed(123)
    b = torch.randn(4)
    assert torch.equal(a, b)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok:", name)
