import argparse
import json
import os

import torch

from tokenizer.tokenizer import CharTokenizer
from training.dataset import get_batch
from model.model import TinyGPT

# ---- hyperparameters ----
CFG = dict(
    batch_size=32,
    block_size=64,     # context length
    n_embd=128,        # embedding size
    n_head=4,          # attention heads (n_embd must divide by n_head)
    n_layer=4,         # transformer blocks
    dropout=0.1,
    lr=1e-3,
    max_steps=5000,
    eval_interval=500,
    eval_iters=50,
)

parser = argparse.ArgumentParser()
parser.add_argument("--max-steps", type=int, default=CFG["max_steps"])
parser.add_argument("--data", default="data/training.txt")
parser.add_argument("--out", default="checkpoints/tinygpt.pt")
args = parser.parse_args()
CFG["max_steps"] = args.max_steps

torch.manual_seed(1337)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

# ---- data ----
text = open(args.data, encoding="utf-8").read()
tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
splits = {"train": data[:n], "val": data[n:]}

# ---- model ----
model_cfg = dict(
    vocab_size=tok.vocab_size,
    block_size=CFG["block_size"],
    n_embd=CFG["n_embd"],
    n_head=CFG["n_head"],
    n_layer=CFG["n_layer"],
    dropout=CFG["dropout"],
)
model = TinyGPT(**model_cfg).to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])


@torch.no_grad()
def estimate_loss():
    """Average loss over many batches -> far less noisy than a single batch."""
    model.eval()
    out = {}
    for name, d in splits.items():
        losses = torch.zeros(CFG["eval_iters"])
        for k in range(CFG["eval_iters"]):
            xb, yb = get_batch(d, CFG["block_size"], CFG["batch_size"], device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


# ---- training loop ----
history = []  # (step, train_loss, val_loss)
for step in range(CFG["max_steps"]):
    if step % CFG["eval_interval"] == 0 or step == CFG["max_steps"] - 1:
        l = estimate_loss()
        history.append((step, l["train"], l["val"]))
        print(f"step {step}: train loss {l['train']:.4f}, val loss {l['val']:.4f}")

    xb, yb = get_batch(splits["train"], CFG["block_size"], CFG["batch_size"], device)
    _, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

# ---- save checkpoint ----
os.makedirs(os.path.dirname(args.out), exist_ok=True)
torch.save({"model": model.state_dict(), "config": model_cfg, "chars": tok.chars}, args.out)
with open("checkpoints/history.json", "w") as f:
    json.dump(history, f)
print("saved", args.out)

# ---- loss curve (saved as an image for the README) ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [h[0] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, [h[1] for h in history], label="train")
    plt.plot(steps, [h[2] for h in history], label="val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.title("TinyGPT training")
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/loss_curve.png", dpi=150, bbox_inches="tight")
    print("saved assets/loss_curve.png")
except ImportError:
    pass

# ---- sample ----
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(context, max_new_tokens=500)
print(tok.decode(out[0].tolist()))