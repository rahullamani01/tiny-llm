import torch
from tokenizer.tokenizer import CharTokenizer
from training.dataset import get_batch
from model.model import AttentionLanguageModel

# hyperparameters
batch_size = 32
block_size = 32
n_embd = 32
max_steps = 5000
lr = 1e-3
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# data
text = open("data/training.txt", encoding="utf-8").read()
tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# model + optimizer
model = AttentionLanguageModel(tok.vocab_size, block_size, n_embd).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(max_steps):
    xb, yb = get_batch(train_data, block_size, batch_size, device)
    _, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 500 == 0:
        print(f"step {step}: train loss {loss.item():.4f}")

# validation loss
model.eval()
with torch.no_grad():
    xv, yv = get_batch(val_data, block_size, 256, device)
    _, vloss = model(xv, yv)
print(f"val loss {vloss.item():.4f}")

# generate
context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(context, max_new_tokens=300)
print(tok.decode(out[0].tolist()))