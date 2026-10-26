"""Sanity checks for sampling and generation. Run with: python -m tests.test_generation"""
import torch

from inference.generate import generate_text, stream_text
from model.model import TinyGPT
from model.sampling import sample_next_token
from tokenizer.tokenizer import CharTokenizer


def _toy_model(vocab_size=20, block_size=16):
    torch.manual_seed(0)
    return TinyGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=32,
        n_head=4,
        n_layer=2,
    ).eval()


def test_greedy_is_argmax():
    logits = torch.tensor([[0.1, 3.0, 2.0]])
    out = sample_next_token(logits, temperature=0.0)
    assert out.shape == (1, 1)
    assert out.item() == 1


def test_top_k_one_is_argmax():
    logits = torch.tensor([[0.1, 3.0, 2.0]])
    out = sample_next_token(logits, temperature=1.0, top_k=1)
    assert out.item() == 1


def test_top_p_keeps_only_the_nucleus():
    logits = torch.tensor([[0.1, 3.0, 2.0]])
    out = sample_next_token(logits, temperature=1.0, top_p=0.01)
    assert out.item() == 1


def test_top_p_always_keeps_one_token():
    logits = torch.zeros(2, 5)  # uniform distribution
    out = sample_next_token(logits, top_p=1e-6)
    assert out.shape == (2, 1)
    assert torch.all((out >= 0) & (out < 5))


def test_repetition_penalty_can_flip_the_winner():
    logits = torch.tensor([[2.0, 1.9, 1.8]])
    generated = torch.tensor([[0]])
    out = sample_next_token(
        logits,
        temperature=0.0,
        repetition_penalty=2.0,
        generated=generated,
    )
    assert out.item() == 1  # token 0 is divided down to 1.0, below 1.9


def test_batched_greedy():
    logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 1.0, 2.0]])
    out = sample_next_token(logits, temperature=0.0)
    assert out.shape == (2, 1)
    assert out.flatten().tolist() == [1, 0]


def test_invalid_options_raise():
    logits = torch.zeros(1, 5)
    bad = [
        dict(temperature=-1.0),
        dict(top_k=0),
        dict(top_p=0.0),
        dict(top_p=1.5),
        dict(repetition_penalty=0.0),
    ]
    for kwargs in bad:
        try:
            sample_next_token(logits, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_generate_is_deterministic_when_greedy():
    model = _toy_model()
    prompt = torch.zeros((1, 1), dtype=torch.long)
    a = model.generate(prompt, 10, temperature=0.0)
    b = model.generate(prompt, 10, temperature=0.0)
    assert a.shape == (1, 11)
    assert torch.equal(a, b)

    with torch.no_grad():
        logits, _ = model(prompt)
    assert a[0, 1].item() == logits[0, -1].argmax().item()


def test_generate_stream_matches_generate():
    model = _toy_model()
    prompt = torch.zeros((1, 1), dtype=torch.long)
    torch.manual_seed(1)
    expected = model.generate(prompt, 10, temperature=0.7, top_k=5)
    torch.manual_seed(1)
    chunks = list(model.generate_stream(prompt, 10, temperature=0.7, top_k=5))
    assert len(chunks) == 10
    assert torch.equal(chunks[-1], expected)


def test_generate_zero_new_tokens():
    model = _toy_model()
    prompt = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(prompt, 0, temperature=0.0)
    assert torch.equal(out, prompt)


def test_stream_text_matches_generate_text():
    tok = CharTokenizer("abc")
    model = _toy_model(vocab_size=tok.vocab_size)

    chunks = list(stream_text(model, tok, "a", 8, temperature=0.0))
    assert len(chunks) == 8
    assert all(len(c) == 1 and c in "abc" for c in chunks)
    assert "".join(chunks) == generate_text(model, tok, "a", 8, temperature=0.0)[1:]


def test_tokenizer_reports_missing_chars():
    tok = CharTokenizer("abc")
    assert tok.missing_chars("cab") == []
    assert tok.missing_chars("cabx") == ["x"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok:", name)
