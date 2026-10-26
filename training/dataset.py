import torch


def split_data(data, val_fraction=0.1):
    """Chronologically split a 1D tensor into train and validation parts."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    n = int((1.0 - val_fraction) * len(data))
    if n <= 0 or n >= len(data):
        raise ValueError("dataset is too small for the requested split")
    return {"train": data[:n], "val": data[n:]}


def get_batch(data, block_size, batch_size, device):
    if len(data) <= block_size:
        raise ValueError("data is shorter than block_size + 1 tokens")
    # pick random starting points in the text
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])          # inputs
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # targets: shifted by 1
    return x.to(device), y.to(device)
