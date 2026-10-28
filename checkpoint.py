"""Save and load TinyGPT checkpoints."""
import os

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
    """Write a resumable checkpoint, creating parent directories as needed."""
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
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint, including legacy files without optimizer state."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only argument
        return torch.load(path, map_location=device)
