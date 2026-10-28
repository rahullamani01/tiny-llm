# KV caching

Autoregressive decoding without a cache re-runs the whole prefix on every
step, so it recomputes keys and values for tokens that were already processed.
The optional KV cache stores each layer's keys and values once, then attends
over them while projecting only the newest token.

Caching is **off by default**; uncached generation and existing checkpoints are
unchanged.

## Forward path

```
idx [B, T]
  -> Embeddings: token table + learned absolute positions pos_offset..pos_offset+T-1
  -> Block x N
       ln1 -> MultiHeadAttention (H heads, each with its own q/k/v = head_size)
           -> concat heads -> proj -> + residual
       ln2 -> FeedForward (4x) -> + residual
  -> ln_f -> lm_head -> logits [B, T, vocab]
```

Files:

- `model/attention.py` — `Head._attend` takes `past_kv` + `pos_offset`; `MultiHeadAttention.forward_with_cache` stacks per-head caches.
- `model/embeddings.py` — `pos_offset` shifts the learned position embedding.
- `model/transformer.py` — `Block.forward_with_cache` passes the cache through.
- `model/model.py` — `TinyGPT.forward(..., past_key_values=..., use_cache=...)` and the cached generation loop.
- `inference/generate.py`, `inference/repl.py` — `--kv-cache` switch.

## Cache layout

One cache per transformer layer (each layer has different weights):

```text
past_key_values: tuple of n_layer items
  each item: (keys, values)
    keys:   [batch, n_head, cached_tokens, head_size]
    values: [batch, n_head, cached_tokens, head_size]
```

At a decode step with one new token, the new keys/values are
`[batch, n_head, 1, head_size]`; they are concatenated along the
`cached_tokens` dimension. Attention weights for that step have shape
`[batch, 1, cached_tokens + 1]`.

The cache is a plain tuple of tensors, so it lives on the same device and uses
the same dtype as the model.

## Causal masking with a cache

The query for the new token sits at absolute position `pos_offset`, so it may
attend to every cached key plus itself. The mask therefore uses row
`pos_offset` of the causal triangle instead of `tril[:1, :T]`:

```python
mask = self.tril[pos_offset:pos_offset + T, :T_k]
```

For example, decoding token 7 of a sequence uses row 7, which allows keys
0 through 7. Using `tril[:1, :T_k]` would only allow key 0 and is a common bug.

## Prefill and decoding

```python
# prefill: process the prompt once and keep the cache
logits, _, past = model(idx, use_cache=True)

# decode: only the newest token goes through the model
logits, _, past = model(next_id, past_key_values=past, use_cache=True)
```

`TinyGPT.forward` returns `(logits, loss)` as before; with `use_cache=True` it
returns `(logits, loss, past_key_values)`. `model.generate(...)` and
`model.generate_stream(...)` accept `use_cache=True` and preserve all sampling
options (temperature, top-k, top-p, repetition penalty, streaming).

## Context-window policy

This model uses **learned absolute** position embeddings, so position `t` must
stay below `block_size`. Its uncached generator already slides the window:

```python
idx_cond = idx[:, -self.block_size:]
```

When the window slides, the retained tokens are re-indexed from position 0.
Their cached keys/values were built with older positions, so reusing them
directly would be wrong. The cached path therefore **rebuilds the cache from
the retained window** whenever the sequence would exceed `block_size`:

```python
window = idx[:, -self.block_size:]
logits, _, past = model(window, use_cache=True)
```

Consequences:

- Cached and uncached outputs match at every length, including past the window.
- Positions never exceed the embedding table; a direct call that would exceed
  it raises `ValueError` mentioning `block_size`.
- Past the window the rebuild costs the same as the uncached recompute, so the
  speedup only applies while `prompt + generated - 1 <= block_size`.

A future RoPE implementation can avoid the rebuild, because rotating cached
keys makes the positions relative.

## Enabling it

CLI:

```bash
python -m inference.generate --prompt $'ROMEO:\n' --kv-cache
python -m inference.generate --prompt $'ROMEO:\n' --kv-cache --stream
python -m inference.repl --kv-cache
```

Python:

```python
from inference.generate import load_model, generate_text

model, tok = load_model()
text = generate_text(model, tok, "ROMEO:\n", use_cache=True)
```

Keep the model in `eval()` mode; dropout during cached decoding would not match
the uncached computation.

## Verifying correctness

```bash
python -m tests.test_cache
```

The tests cover: cached vs uncached logits, per-layer cache shapes, batch > 1,
single-token prompts, greedy equality across the context boundary, streaming,
cache-disabled behavior, and the out-of-range position errors.

## Benchmark

```bash
python -m experiments.kv_cache_benchmark
python -m experiments.kv_cache_benchmark --prompt-tokens 128 --tokens 128
python -m experiments.kv_cache_benchmark --ckpt checkpoints/tinygpt.pt --tokens 16
```

The script uses the same weights, device, dtype, prompt and token budget for
both paths, verifies that the greedy outputs match, and reports prefill/decode/
end-to-end times, tokens per second, the speedup ratio, and cache size. It
refuses workloads beyond the context window because that regime rebuilds every
step and has no decode speedup to measure. For GPU runs it synchronizes around
timed sections and reports CUDA peak memory; CPU runs report the median of
`--runs` repeats.

### Measured example

Apple M5, 16 GB, macOS 26.4.1, Python 3.14, PyTorch 2.14.1, float32, CPU,
median of 3 runs, random 4-layer / 4-head / 128-dim model with `block_size=256`:

```text
Prompt tokens: 32   Generated tokens: 32
                  Uncached      Cached
prefill time (s)     0.001       0.001
decode time (s)      0.040       0.023
end-to-end (s)       0.042       0.024
decode tokens/s      793.1      1381.5
end-to-end tokens/s  771.1      1310.6

speedup: 1.70x
KV cache: 0.25 MiB (63 cached tokens)
greedy output match: yes
```

With a 128-token prompt and 128 generated tokens the same setup reached
`3.22x` (decode 354 -> 1159 tokens/s, cache 1.00 MiB). Short sequences and small
models benefit less: on MPS with batch 4, 16+16 tokens, caching was roughly at
parity because per-step overhead is small compared with the tiny attention
matmuls. Measure your own setup rather than assuming a fixed speedup.

## Cache memory

```text
bytes = 2 * layers * batch * num_kv_heads * cached_tokens * head_dim * bytes_per_element
```

For the 32-token example: `2 * 4 * 1 * 4 * 63 * 32 * 4 = 258,048` bytes
(0.25 MiB), matching the measured tensor size. This excludes allocator
overhead, activations, and logits. Memory grows linearly with context length
and batch size, which is the main tradeoff of caching.

## Limitations

- Standard multi-head attention only; no grouped-query attention yet.
- No quantized/paged caches, continuous batching, or speculative decoding.
- The cache is rebuilt when the learned-absolute window slides, trading some
  speed for exact equivalence with the uncached generator.
- Cache tensors are not serialized in checkpoints; they are created fresh per
  generation.

## Study questions

1. Why must Q be recomputed for new tokens while old K/V can be reused?
2. Why does every transformer layer need its own cache?
3. Why do prefill and decoding have different compute patterns?
4. Why does the mask need row `pos_offset` rather than row 0?
5. Why does cache memory grow with context length?
6. How would RoPE or grouped-query attention change the cache design?
