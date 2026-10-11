# Later: compare generation at different temperatures
"""Generate from the same prompt at different temperatures and compare."""
import torch

from inference.generate import load_model, generate_text

model, tok = load_model()
prompt = "ROMEO:\n"

for temp in [0.3, 0.7, 1.0, 1.5]:
    torch.manual_seed(0)  # same randomness each time, so only temperature changes
    print(f"\n===== temperature {temp} =====")
    print(generate_text(model, tok, prompt, max_new_tokens=300, temperature=temp, top_k=None))