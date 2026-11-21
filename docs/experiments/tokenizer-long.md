# Longer tokenizer comparison

## Setup

We ran three seeds (42, 1337, 2026) on the same chronological raw-text split
and the small architectures described in the
[baseline](tokenizer-baseline.md). Each model trained for 1,000 updates,
batch size 8 and context 64; 40 random evaluation batches were measured every
200 steps. Evaluation uses the same fixed sampling seed at each point and
across training seeds. The tokenizer and model sizes differ; equal numbers of
token updates do **not** mean equal compute or equal characters seen.

Reproduce one seed with a fresh output path:

```bash
python -m experiments.tokenizer_baseline --device auto --seed 1337 \
  --steps 1000 --eval-interval 200 --eval-iters 40 --sample-tokens 80 \
  --output checkpoints/tokenizer-long-1337.json
```

Full configuration, curves and samples:
[seed 42](tokenizer-long-42.json),
[seed 1337](tokenizer-long-1337.json),
[seed 2026](tokenizer-long-2026.json). Runs used Apple M5 / MPS and PyTorch
2.14.1. They save metrics and samples, not model weights.

## Common-unit metric

For each sampled target token, we add its negative log probability and count
the UTF-8 bytes in its decoded text. Bits per byte is total negative log
probability divided by `ln(2) * target bytes`; lower is better. The metric is
an estimate on random windows, not a whole-dataset score. The test suite checks
the byte count with a multibyte character.

This normalization makes the numbers more interpretable across tokenizers,
but it is not a perfectly controlled comparison: tokenizers sample different
windows and expose different amounts of text in a 64-token context. BPE scores
the chosen deterministic tokenization, not the exact marginal probability of
the text over every possible token sequence. Evaluation also uses overlapping
windows and does not predict the first token of each window.

## Results

Validation bits per byte; each cell is the range across the three seeds:

| Steps | Character | BPE |
|---:|---:|---:|
| 200 | 3.630–3.634 | 3.195–3.221 |
| 400 | 3.497–3.515 | 3.041–3.065 |
| 600 | 3.358–3.426 | 2.975–2.986 |
| 800 | 3.254–3.279 | 2.923–2.932 |
| 1,000 | 3.181–3.201 | 2.880–2.884 |

In this setup BPE has a lower sampled normalized loss at every measured step
and seed, and both improve throughout. This does not prove BPE is universally
better: its model has more parameters, its context covers more text, and it
sees more characters per training update.

The fixed greedy prompts `ROMEO:` and `JULIET:` also expose a limitation:
character models frequently repeat “the”; BPE models produce longer
Shakespeare-like fragments but still repeat and often invent words. Samples
are limited to 80 *tokens*, so the BPE completions are longer in characters.
These tiny models are not reliable conversational assistants.

For a stronger follow-up, evaluate identical contiguous text spans, compare
matched compute and parameter budgets, add sampled as well as greedy
completions, and test on another dataset.
