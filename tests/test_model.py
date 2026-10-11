"""Quick sanity checks. Run with:  python -m tests.test_model"""
import torch

from tokenizer.tokenizer import CharTokenizer
from model.model import TinyGPT


def test_tokenizer_roundtrip():
    tok = CharTokenizer("hello world")
    assert tok.decode(tok.encode("hello")) == "hello"
    assert CharTokenizer.from_chars(tok.chars).chars == tok.chars


def test_output_shapes_and_loss():
    model = TinyGPT(vocab_size=20, block_size=16, n_embd=32, n_head=4, n_layer=2)
    x = torch.randint(0, 20, (3, 16))
    logits, loss = model(x, x)
    assert logits.shape == (3, 16, 20)
    assert loss.item() > 0


def test_causal_mask_no_future_leak():
    torch.manual_seed(0)
    model = TinyGPT(vocab_size=20, block_size=16, n_embd=32, n_head=4, n_layer=2).eval()
    x = torch.randint(0, 20, (1, 16))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % 20          # change only the LAST token
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(y)
    # all earlier positions must be unaffected by a change in the future
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-5)


def test_generate_length():
    model = TinyGPT(vocab_size=20, block_size=16, n_embd=32, n_head=4, n_layer=2).eval()
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), 40, temperature=0.8, top_k=5)
    assert out.shape == (1, 41)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok:", name)