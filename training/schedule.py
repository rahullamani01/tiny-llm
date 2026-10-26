"""Learning-rate schedules."""
import math


def cosine_lr(step, max_steps, base_lr, warmup_steps=0, min_lr=0.0):
    """Linear warmup followed by cosine decay from base_lr down to min_lr."""
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if step < 0:
        raise ValueError("step must be >= 0")
    if base_lr < 0 or min_lr < 0:
        raise ValueError("learning rates must be >= 0")
    if min_lr > base_lr:
        raise ValueError("min_lr must be <= base_lr")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if warmup_steps > max_steps:
        raise ValueError("warmup_steps must be <= max_steps")

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    progress = min(max(step - warmup_steps, 0), max_steps - warmup_steps) / max(
        1, max_steps - warmup_steps
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)


def get_lr(step, max_steps, base_lr, schedule="cosine", warmup_steps=0, min_lr=0.0):
    """Return the learning rate for a step; "none" keeps base_lr constant."""
    if schedule == "none":
        if base_lr < 0:
            raise ValueError("base_lr must be >= 0")
        return base_lr
    if schedule == "cosine":
        return cosine_lr(step, max_steps, base_lr, warmup_steps, min_lr)
    raise ValueError(f"unknown lr schedule {schedule!r}")
