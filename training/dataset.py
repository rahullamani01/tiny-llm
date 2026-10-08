import torch


def get_batch(data, block_size, batch_size, device):
    # pick random starting points in the text
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])          # inputs
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # targets: shifted by 1
    return x.to(device), y.to(device)
