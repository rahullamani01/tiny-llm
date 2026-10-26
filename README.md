# tiny-llm

A tiny GPT built from scratch in PyTorch to learn how LLMs work.

## Roadmap
- [x] Character tokenizer
- [x] Bigram baseline
- [x] Embeddings + position information
- [x] Self-attention
- [x] Multi-head attention
- [x] Transformer block
- [x] Stacked blocks -> Tiny GPT
- [x] Training
- [x] Text generation (temperature, top-k, top-p, repetition penalty, streaming)

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/download.py
python -m training.train
```

## Generate text

One-shot generation (defaults: `temperature=0.8`, `top-k=40`):
```bash
python -m inference.generate --prompt $'ROMEO:\n' --tokens 300
```

Sampling knobs:
```bash
# nucleus sampling instead of top-k
python -m inference.generate --prompt $'ROMEO:\n' --top-p 0.9 --top-k 0

# greedy / deterministic
python -m inference.generate --prompt $'ROMEO:\n' --temperature 0

# discourage repeats and make runs reproducible
python -m inference.generate --prompt $'ROMEO:\n' --repetition-penalty 1.1 --seed 0

# print each character as soon as it is sampled
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
python -m tests.test_model
python -m tests.test_generation
```

## Project layout
```
data/         training text
tokenizer/    char tokenizer, vocabulary
model/        embeddings, attention, transformer, model, sampling
training/     dataset batching, training loop
inference/    text generation, streaming, interactive REPL
experiments/  attention visualization, temperature test
checkpoints/  saved weights (git-ignored)
```
