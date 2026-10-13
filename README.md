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
- [x] Text generation (temperature, top-k)

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/download.py
python -m training.train
```

## Project layout
```
data/         training text
tokenizer/    char tokenizer, vocabulary
model/        embeddings, attention, transformer, model
training/     dataset batching, training loop
inference/    text generation
experiments/  attention visualization, temperature test
checkpoints/  saved weights (git-ignored)
```
