# Learning roadmap

This is a personal learning project, not a promise of a production-quality
assistant. Work in small experiments: write down a question, change one thing,
measure it, and record what you learned.

## Stage 1: Understand and establish a baseline

1. Trace text through `tokenizer/`, `training/dataset.py`, `model/`, and
   `inference/`. Explain the shape of the tensors at each step.
2. Run `python -m tests.run_all` and the reference training recipe in the
   README. Start with a small run if compute is limited.
3. Record the dataset, seed, device, model configuration, training steps,
   validation loss, runtime, and a few fixed-prompt samples. Save checkpoints
   separately for each run.
4. Read `training/evaluate.py`: loss measures next-token prediction, while
   generated samples offer a different, qualitative view.

## Stage 2: Controlled experiments

Compare character and BPE tokenizers, then try a change to context length or
model size. Keep the data split and evaluation procedure consistent. Record
resource costs as well as loss; a larger model is not automatically better.
Evaluation on the same text used for training is not a substitute for validation.

## Stage 3: Better data and chat

Inspect instruction examples and their licenses, remove duplicates, and keep
training and evaluation conversations separate. Compare the base model and SFT
model on a small, fixed set of prompts. Document failures instead of relying on
one impressive sample.

## Stage 4: Systems and deeper research

Profile inference, investigate KV caching and quantization, and explore retrieval
or alternative architectures when there is a clear question to answer. Add tests
and explain tradeoffs for each change.

## Experiment notes template

```text
Question:
Hypothesis:
Dataset and split:
Configuration, seed, device, and code revision:
Command:
Runtime and resource use:
Train/validation loss:
Fixed prompts and samples:
Conclusion and next question:
```

Seeds help comparisons, but exact reproducibility is not guaranteed across
hardware, PyTorch versions, or resumed runs.
