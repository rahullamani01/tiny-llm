"""Save and load TinyGPT checkpoints."""
import os
import tempfile

import torch

FORMAT_VERSION = 2


def save_checkpoint(
    path,
    model,
    optimizer,
    model_config,
    chars,
    step,
    history,
    best_val_loss,
    train_config=None,
    tokenizer=None,
):
    """Atomically replace a resumable checkpoint after a successful write."""
    payload = {
        "version": FORMAT_VERSION,
        "model": model.state_dict(),
        "config": dict(model_config),
        "chars": list(chars),
        "step": int(step),
        "history": list(history),
        "best_val_loss": float(best_val_loss),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if train_config is not None:
        payload["train_config"] = dict(train_config)
    if tokenizer is not None:
        payload["tokenizer"] = dict(tokenizer)

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # Write beside the destination so os.replace stays on the same filesystem.
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=parent or ".")
    try:
        with os.fdopen(fd, "wb") as file:
            torch.save(payload, file)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, device="cpu"):
    """Load a trusted checkpoint, including legacy files without optimizer state.

    PyTorch pickle loading can execute code: never load untrusted checkpoints.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only argument
        return torch.load(path, map_location=device)
