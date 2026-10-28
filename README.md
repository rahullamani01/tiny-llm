# tiny-llm

[![tests](https://github.com/rahullamani01/tiny-llm/actions/workflows/tests.yml/badge.svg)](https://github.com/rahullamani01/tiny-llm/actions/workflows/tests.yml)

A tiny GPT built from scratch in PyTorch to learn how LLMs work.

## Roadmap
- [x] Character tokenizer
- [x] BPE tokenizer (optional)
- [x] Bigram baseline
- [x] Embeddings + position information
- [x] Self-attention
- [x] Multi-head attention
- [x] Transformer block
- [x] Stacked blocks -> Tiny GPT
- [x] Training
- [x] Text generation (temperature, top-k, top-p, repetition penalty, streaming)
- [x] Cosine LR schedule, warmup, gradient clipping
- [x] Resumable training and best-checkpoint tracking
- [x] Evaluation (loss and perplexity)
- [x] Optional weight tying
- [x] Continuous integration

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/download.py
python -m training.train
```

## Training

Default run (5000 steps, cosine schedule with 100-step warmup, gradient clipping):
```bash
python -m training.train
```

Useful options:
```bash
# smaller/faster run
python -m training.train --max-steps 2000 --batch-size 16 --lr 5e-4

# share the token embedding and output projection
python -m training.train --tie-weights

# disable the LR schedule or force a device
python -m training.train --lr-schedule none --device cpu

# extend a finished run
python -m training.train --resume --max-steps 8000
```

Checkpoints are written next to `--out` (default `checkpoints/tinygpt.pt`):
- `tinygpt.pt` — latest model plus optimizer/step/history, used by `--resume`
- `best.pt` — best validation loss seen so far
- `history.json` — loss history used for the training curve

### BPE tokenizer

The default tokenizer is character-level. A BPE vocabulary is optional and
usually gives ~2x shorter sequences on Shakespeare:

```bash
# learn 512 tokens (65 characters + 447 merges)
python -m tokenizer.train_bpe --vocab-size 512 --out checkpoints/bpe.json

# train with it
python -m training.train --tokenizer bpe --vocab-file checkpoints/bpe.json \
    --out checkpoints/tinygpt-bpe.pt

# resume with the same vocabulary
python -m training.train --resume --tokenizer bpe --vocab-file checkpoints/bpe.json \
    --out checkpoints/tinygpt-bpe.pt --max-steps 8000
```

If `--vocab-file` does not exist, training learns one from the corpus and saves
it next to `--out`. `--bpe-max-chars` controls how much text is used to learn
merges (default 100k). Checkpoints store the tokenizer state, so generation and
evaluation load the right one automatically.

Resuming requires the same data/vocabulary and architecture; pass `--tie-weights`
again if the original run used it, and pass the same `--tokenizer`/`--vocab-file`
for BPE runs. The LR schedule is recalculated from `--max-steps`, so extending a
run starts a fresh decay from the current step.

## Evaluate
```bash
python -m inference.evaluate
python -m inference.evaluate --split train --iters 200
python -m inference.evaluate --ckpt checkpoints/best.pt --batch-size 16
```

This prints cross-entropy loss and perplexity on the chosen split.

## Generate text

One-shot generation (defaults: `temperature=0.8`, `top-k=40`):
```bash
python -m inference.generate --prompt $'ROMEO:\n' --tokens 300
```

Sampling knobs:
```bash
# use a BPE checkpoint
python -m inference.generate --ckpt checkpoints/tinygpt-bpe.pt --prompt $'ROMEO:\n'

# nucleus sampling instead of top-k
python -m inference.generate --prompt $'ROMEO:\n' --top-p 0.9 --top-k 0

# greedy / deterministic
python -m inference.generate --prompt $'ROMEO:\n' --temperature 0

# discourage repeats and make runs reproducible
python -m inference.generate --prompt $'ROMEO:\n' --repetition-penalty 1.1 --seed 0

# print each token as soon as it is sampled
python -m inference.generate --prompt $'ROMEO:\n' --stream
```

Interactive streaming REPL — type a prompt, the model continues the running
context. Use `/help` to change the sampling settings live, `/reset` to start
over, and `/quit` to exit:
```bash
python -m inference.repl
```

## Tests
```bash
python -m tests.run_all
```

The same suite runs in GitHub Actions on every push and pull request.

## Project layout
```
data/         training text
tokenizer/    char + BPE tokenizers, vocabulary state
model/        embeddings, attention, transformer, model, sampling
training/     dataset batching, schedules, evaluation, training loop
inference/    text generation, streaming, REPL, evaluation CLI
experiments/  attention visualization, temperature test
checkpoints/  saved weights (git-ignored)
utils.py      device selection and seeding
checkpoint.py checkpoint save/load
```
