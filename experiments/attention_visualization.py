"""Plot the attention patterns of every head in one layer."""
import argparse
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inference.generate import load_model

parser = argparse.ArgumentParser(description="Plot attention weights for one layer.")
parser.add_argument("layer", nargs="?", type=int, default=0, help="layer index (default: 0)")
parser.add_argument("--ckpt", default="checkpoints/tinygpt.pt", help="checkpoint path")
parser.add_argument("--out", default="assets/attention.png", help="output image path")
args = parser.parse_args()

text = "First Citizen:\nBefore we proceed any further"

model, tok = load_model(args.ckpt)
if not 0 <= args.layer < len(model.blocks):
    raise SystemExit(f"error: layer must be in 0..{len(model.blocks) - 1}")

device = next(model.parameters()).device
ids = tok.encode(text)[: model.block_size]
tokens = [tok.decode([i]).replace("\n", "\\n") for i in ids]

with torch.no_grad():
    model(torch.tensor([ids], device=device))

heads = model.blocks[args.layer].sa.heads
fig, axes = plt.subplots(1, len(heads), figsize=(4.5 * len(heads), 4.5), squeeze=False)
for i, (ax, head) in enumerate(zip(axes[0], heads)):
    ax.imshow(head.last_attn[0].cpu(), cmap="viridis")
    ax.set_title(f"layer {args.layer} head {i}")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=6)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=6)
    ax.set_xlabel("attended to")
    ax.set_ylabel("current token")

plt.tight_layout()
parent = os.path.dirname(args.out)
if parent:
    os.makedirs(parent, exist_ok=True)
plt.savefig(args.out, dpi=150)
print(f"saved {args.out}")
