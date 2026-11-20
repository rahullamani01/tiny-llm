"""Loss evaluation shared by the training loop and the evaluation CLI."""
import torch

from training.dataset import get_batch


def evaluate_loss(
    model,
    data,
    block_size,
    batch_size=32,
    iters=100,
    device=None,
    seed=None,
):
    """Average cross-entropy over random batches without changing model mode.

    A seed uses a local random generator, leaving the training RNG untouched.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if iters < 1:
        raise ValueError("iters must be >= 1")
    if len(data) <= block_size:
        raise ValueError("data is shorter than block_size + 1 tokens")

    device = device or next(model.parameters()).device
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    was_training = model.training
    model.eval()
    total = 0.0
    try:
        with torch.no_grad():
            for _ in range(iters):
                xb, yb = get_batch(data, block_size, batch_size, device, generator=generator)
                _, loss = model(xb, yb)
                total += loss.item()
    finally:
        model.train(was_training)
    return total / iters
