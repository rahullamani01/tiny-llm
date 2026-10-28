# Rotary position embeddings (RoPE)

`--pos-embedding rope` trains a model that encodes positions by rotating each
head's queries and keys instead of adding a learned position vector. The
default remains `learned`, so existing checkpoints and runs are unchanged.

## Why rotate?

A learned position table adds a different vector to every absolute position.
Attention then sees where a token is, but the score between two tokens does not
depend only on their distance. RoPE instead rotates a query by its position and
a key by its position; their dot product is a function of the **relative**
offset.

For every adjacent pair `(x[2i], x[2i+1])` in a head vector:

```text
angle_i(pos) = pos / base^(2i / head_size)
even' = even * cos(angle_i) - odd * sin(angle_i)
odd'  = even * sin(angle_i) + odd * cos(angle_i)
```

with `base = 10_000`. Pairs rotate at different frequencies, giving a
multi-scale position signal. This is the GPT-NeoX interleaved-pair layout;
some implementations split the vector into halves instead.

## Shapes and code path

```text
x [B, T, n_embd]
  -> Embeddings: token embedding only (no position table)
  -> per head: q, k, v [B, T, head_size]
       q = rope(q, pos_offset)
       k = rope(k, pos_offset)
       append k/v to the KV cache
  -> attention scores use rotated q/k
```

`model/rope.py` precomputes `cos`/`sin` tables for every position in
`block_size`. The tables are non-persistent buffers, so they do not appear in
checkpoints and are rebuilt on load. No new parameters are introduced.

## Relative property

```python
rope(q, 7) @ rope(k, 3)  ==  rope(q, 27) @ rope(k, 23)
```

Both pairs share the same offset `4`, so the score is identical (up to floating
point). The tests check this, plus norm preservation and the identity at
position 0. Because cached keys are already rotated at their original absolute
positions, decoding one new token only needs to rotate its query and key.

## Training

RoPE changes the architecture, so train from scratch rather than resuming a
learned-position checkpoint:

```bash
python -m training.train --pos-embedding rope \
    --out checkpoints/tinygpt-rope.pt --best-out checkpoints/best-rope.pt
```

The choice is stored in the checkpoint config. Inference and evaluation load it
automatically:

```bash
python -m inference.generate --ckpt checkpoints/tinygpt-rope.pt --prompt $'ROMEO:\n'
python -m inference.generate --ckpt checkpoints/tinygpt-rope.pt --kv-cache
python -m inference.evaluate --ckpt checkpoints/tinygpt-rope.pt
```

Resuming requires the same `--pos-embedding`. A mismatch is reported instead of
silently loading incompatible weights:

```text
error: checkpoint pos_embedding='rope' does not match current pos_embedding='learned'
```

RoPE also requires an even `head_size` (`n_embd / n_head`), because dimensions
are rotated in pairs.

## Interaction with the KV cache

`docs/kv_cache.md` describes the cache layout. With RoPE:

- Cached keys were rotated when they were first computed; they are stored
  rotated.
- The mask still uses absolute positions (`pos_offset`), which is consistent
  because rotation preserves the attention scores for the same relative layout.
- When the context window slides, this implementation still rebuilds the cache
  from the retained `block_size` tokens and restarts positions at 0. That keeps
  cached and uncached output identical for both position schemes.

A natural next optimization is to keep a rolling cache in RoPE mode: drop the
oldest keys/values and continue at absolute positions, since scores only depend
on distance. That avoids the rebuild past the window and is a good exercise, but
it changes the generation bookkeeping and is intentionally left out here.

Both paths can be compared with:

```bash
python -m experiments.kv_cache_benchmark --pos-embedding rope \
    --prompt-tokens 64 --tokens 64
```

## Learned vs RoPE

| | learned absolute | RoPE |
|---|---|---|
| Position signal | added vector table | rotation of q/k |
| Parameters | `block_size * n_embd` | none |
| Score depends on | absolute positions | relative offset |
| Works with KV cache | yes, via position offset | yes, keys stored rotated |
| Old checkpoints | default | requires `--pos-embedding rope` |

## Tests

```bash
python -m tests.test_rope
```

Covered: rotation norms, identity at position 0, relative-score invariance,
distance sensitivity, absence/presence of the position table, cached vs
uncached logits, greedy equality across the context window, a backward pass,
checkpoint round-trip, odd head sizes, and unknown schemes.

## Limitations

- RoPE must be trained from scratch; learned-position weights cannot be reused.
- The frequency base is fixed at 10,000; it is not exposed as a CLI flag.
- Sliding-window cache reuse is not implemented yet (the cache is rebuilt).
- Standard multi-head attention only; no grouped-query attention.

## Study questions

1. Why does a dot product of two rotated vectors depend on position difference?
2. Why do different dimension pairs use different frequencies?
3. Why can cached keys stay rotated, while the query must be rotated every step?
4. Why is there no position embedding table in a RoPE model?
5. How would a rolling cache work past `block_size` under RoPE?
6. How does RoPE interact with grouped-query attention?
