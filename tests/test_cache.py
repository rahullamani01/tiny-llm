"""Correctness tests for optional KV caching. Run: python -m tests.test_cache"""
import torch

from model.model import TinyGPT


def _toy_model(block_size=16, n_layer=2, n_head=4, n_embd=32, vocab_size=20):
    torch.manual_seed(0)
    return TinyGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
    ).eval()


def _tokens(batch, length, vocab_size=20, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (batch, length), generator=generator)


def test_cached_logits_match_uncached_forward():
    model = _toy_model()
    x = _tokens(batch=2, length=12, seed=1)

    with torch.no_grad():
        full, _ = model(x)
        # Prefill the first 7 tokens, then decode the remaining 5 one at a time.
        first, _, past = model(x[:, :7], use_cache=True)
        pieces = [first]
        for t in range(7, 12):
            logits, _, past = model(
                x[:, t:t + 1], past_key_values=past, use_cache=True
            )
            pieces.append(logits)
        incremental = torch.cat(pieces, dim=1)

    assert full.shape == incremental.shape == (2, 12, 20)
    torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-4)


def test_cache_shapes_and_layers_are_independent():
    model = _toy_model(block_size=16, n_layer=3, n_head=4, n_embd=32)
    x = _tokens(batch=2, length=7, seed=2)
    head_size = 32 // 4

    with torch.no_grad():
        _, _, past = model(x, use_cache=True)
        _, _, past_next = model(x[:, -1:], past_key_values=past, use_cache=True)

    assert isinstance(past, tuple) and len(past) == 3
    for keys, values in past:
        assert keys.shape == (2, 4, 7, head_size)
        assert values.shape == (2, 4, 7, head_size)
    for keys, values in past_next:
        assert keys.shape == (2, 4, 8, head_size)
        assert values.shape == (2, 4, 8, head_size)

    # Each layer owns its own cache tensors.
    assert past[0][0].data_ptr() != past[1][0].data_ptr()
    # A one-token query attends over all 8 cached keys.
    assert model.blocks[0].sa.heads[0].last_attn.shape == (2, 1, 8)


def test_single_token_prompt():
    model = _toy_model()

    with torch.no_grad():
        step0, _, past = model(torch.tensor([[3]]), use_cache=True)
        assert step0.shape == (1, 1, 20)
        assert all(k.shape[2] == 1 for k, _ in past)

        step1, _, past2 = model(
            torch.tensor([[7]]), past_key_values=past, use_cache=True
        )
        full, _ = model(torch.tensor([[3, 7]]))

    torch.testing.assert_close(step1[:, -1], full[:, -1], atol=1e-5, rtol=1e-4)
    assert all(k.shape[2] == 2 for k, _ in past2)


def test_batched_cached_greedy_matches_uncached():
    model = _toy_model()
    prompt = _tokens(batch=3, length=5, seed=3)

    uncached = model.generate(prompt, 10, temperature=0.0, use_cache=False)
    cached = model.generate(prompt, 10, temperature=0.0, use_cache=True)

    assert cached.shape == (3, 15)
    assert torch.equal(cached, uncached)


def test_cached_generation_across_context_window():
    model = _toy_model(block_size=8)
    prompt = _tokens(batch=2, length=3, seed=4)

    uncached = model.generate(prompt, 20, temperature=0.0, use_cache=False)
    cached = model.generate(prompt, 20, temperature=0.0, use_cache=True)

    assert cached.shape == (2, 23)
    # Past the window the cache is rebuilt from the retained tokens, so the
    # greedy sequence must still match the uncached sliding-window generator.
    assert torch.equal(cached, uncached)


def test_generation_exactly_at_context_limit():
    model = _toy_model(block_size=8)
    prompt = _tokens(batch=1, length=4, seed=10)

    uncached = model.generate(prompt, 4, temperature=0.0, use_cache=False)
    cached = model.generate(prompt, 4, temperature=0.0, use_cache=True)

    assert cached.shape == (1, 8)
    assert torch.equal(cached, uncached)


def test_prompt_longer_than_context_is_cropped_consistently():
    model = _toy_model(block_size=8)
    prompt = _tokens(batch=1, length=12, seed=5)

    uncached = model.generate(prompt, 6, temperature=0.0, use_cache=False)
    cached = model.generate(prompt, 6, temperature=0.0, use_cache=True)

    assert torch.equal(cached, uncached)


def test_cached_stream_matches_generate():
    model = _toy_model()
    prompt = _tokens(batch=2, length=4, seed=6)

    torch.manual_seed(7)
    expected = model.generate(prompt, 8, temperature=0.7, top_k=5, use_cache=True)
    torch.manual_seed(7)
    chunks = list(
        model.generate_stream(prompt, 8, temperature=0.7, top_k=5, use_cache=True)
    )

    assert len(chunks) == 8
    assert all(chunk.shape == (2, 5 + i) for i, chunk in enumerate(chunks))
    assert torch.equal(chunks[-1], expected)


def test_cache_disabled_matches_default_path():
    model = _toy_model()
    prompt = _tokens(batch=2, length=6, seed=8)

    default = model.generate(prompt, 6, temperature=0.0)
    disabled = model.generate(prompt, 6, temperature=0.0, use_cache=False)

    assert torch.equal(default, disabled)


def test_cached_zero_new_tokens():
    model = _toy_model()
    prompt = _tokens(batch=1, length=4, seed=9)

    out = model.generate(prompt, 0, temperature=0.0, use_cache=True)
    assert torch.equal(out, prompt)


def test_positions_beyond_block_size_raise():
    model = _toy_model(block_size=8)
    with torch.no_grad():
        _, _, full_cache = model(torch.zeros((1, 8), dtype=torch.long), use_cache=True)

        bad_calls = [
            lambda: model(
                torch.zeros((1, 1), dtype=torch.long),
                past_key_values=full_cache,
                use_cache=True,
            ),
            lambda: model(torch.zeros((1, 9), dtype=torch.long)),
        ]
        for call in bad_calls:
            try:
                call()
            except ValueError as e:
                assert "block_size" in str(e), e
            else:
                raise AssertionError("expected ValueError past the context window")


def test_past_requires_use_cache():
    model = _toy_model()
    try:
        model(torch.zeros((1, 2), dtype=torch.long), past_key_values=())
    except ValueError as e:
        assert "use_cache" in str(e), e
    else:
        raise AssertionError("expected ValueError when past_key_values is passed alone")


def test_cache_rejects_mismatched_batch():
    model = _toy_model()
    with torch.no_grad():
        _, _, past = model(torch.zeros((2, 3), dtype=torch.long), use_cache=True)
        try:
            model(
                torch.zeros((1, 1), dtype=torch.long),
                past_key_values=past,
                use_cache=True,
            )
        except ValueError as e:
            assert "batch" in str(e), e
        else:
            raise AssertionError("expected ValueError for a mismatched batch size")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok:", name)
