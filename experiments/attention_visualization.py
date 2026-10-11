# Later: plot attention weights with matplotlib
"""Plot the attention patterns of every head in one layer."""
import sys

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inference.generate import load_model

text = "First Citizen:\nBefore we proceed any further"
layer = int(sys.argv[1]) if len(sys.argv) > 1 else 0

model, tok = load_model()
device = next(model.parameters()).device
ids = tok.encode(text)[: model.block_size]
chars = [c.replace("\n", "\\n") for c in text[: len(ids)]]

with torch.no_grad():
    model(torch.tensor([ids], device=device))

heads = model.blocks[layer].sa.heads
fig, axes = plt.subplots(1, len(heads), figsize=(4.5 * len(heads), 4.5), squeeze=False)
for i, (ax, head) in enumerate(zip(axes[0], heads)):
    ax.imshow(head.last_attn[0].cpu(), cmap="viridis")
    ax.set_title(f"layer {layer} head {i}")
    ax.set_xticks(range(len(chars)))
    ax.set_xticklabels(chars, fontsize=6)
    ax.set_yticks(range(len(chars)))
    ax.set_yticklabels(chars, fontsize=6)
    ax.set_xlabel("attended to")
    ax.set_ylabel("current token")

plt.tight_layout()
plt.savefig("assets/attention.png", dpi=150)
print("saved assets/attention.png")