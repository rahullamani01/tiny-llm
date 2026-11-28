# UTF-8 byte tokenizer: first smoke run

The previous chat run excluded two validation conversations containing
characters absent from the training vocabulary. The new fixed byte tokenizer
encodes UTF-8 directly, so those examples can be included without building a
vocabulary from validation data. This is a coverage and pipeline smoke test,
not evidence of improved chat quality.

## What changed

Each token ID 0–255 represents one byte; SFT appends the five existing special
tokens. This is not BPE, and may require more tokens for the same text. A
128-token window contains at most 128 bytes, not 128 Unicode characters or
words. Streaming uses an incremental UTF-8 decoder so multi-byte characters
are emitted together; invalid or incomplete generated sequences become `�`.
Unpaired Python surrogate code points are not valid UTF-8 and cannot be encoded.

## Run

We reused the split documented in [chat-phase.md](chat-phase.md), using **all
300** validation conversations rather than the filtered 298. The training set
contained 2,700 conversations. On Apple M5 / MPS with PyTorch 2.14.1 and seed 0,
we trained a 2-layer, 128-dimensional model for 100 base steps and 100 SFT
steps; context 128 bytes, batch 8, four heads, dropout 0.1.

| Stage | Step | Sampled validation loss (nats per byte token) |
|---|---:|---:|
| Base | 0 | 5.5913 |
| Base | 100 | 2.6503 |
| SFT | 0 | 2.6572 |
| SFT | 100 | 2.5889 |

Base validation is a split of the pretraining text; SFT validation is the
held-out conversation set. SFT evaluates only assistant targets, including
special end tokens, so its loss is not strictly bits-per-byte. Evaluation
batches differ across checkpoints. These losses are not directly comparable
with the earlier BPE run, which used different token units, lengths, and
training steps. No claims about answer quality follow from this short run.

Reproduce after preparing the split as in `chat-phase.md`:

```bash
python -m training.train --data data/cache/phase-chat-base.txt \
  --out checkpoints/phase-byte/base.pt --tokenizer byte \
  --block-size 128 --n-embd 128 --n-head 4 --n-layer 2 --batch-size 8 \
  --max-steps 100 --warmup-steps 10 --eval-interval 100 --eval-iters 10 \
  --seed 0 --device mps --sample-tokens 0 --no-plot

python -m training.sft --data data/cache/phase-chat-train.jsonl \
  --val-data data/cache/phase-chat-val.jsonl --init-from checkpoints/phase-byte/base.pt \
  --out checkpoints/phase-byte/chat.pt --block-size 128 --n-embd 128 \
  --n-head 4 --n-layer 2 --batch-size 8 --max-steps 100 --warmup-steps 10 \
  --eval-interval 100 --eval-iters 10 --seed 0 --device mps --no-plot
```

Use new output paths to avoid overwriting runs. Data and checkpoints remain
local and are git-ignored. A future comparison should evaluate generated
answers and use matched compute and text-context budgets.
