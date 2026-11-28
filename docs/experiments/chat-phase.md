# First corrected chat-training run

## Goal and outcome

This is a small end-to-end test of the corrected next-token SFT pipeline, not a
capability claim. The model did not produce useful answers in our fixed-prompt
inspection. Both base and chat outputs were repetitive or nonsensical, and
all ten continuations reached the 64-token limit without a chat stop token.
The full outputs, including checkpoint and prompt hashes, are in
[chat-phase-results.json](chat-phase-results.json).

## Data and setup

- Source: Alpaca dataset, downloaded by `data.prepare_chat` on this machine.
  The downloaded file's SHA-256 was
  `2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419`.
  Dataset sources can change; verify the hash and license before reuse.
- Seed 0; 3,000 converted examples, split into 2,700 training and 300
  validation conversations by normalized prompt groups before pretraining.
- The training character vocabulary lacked `´` and `×` found in the validation
  split. We excluded the two affected conversations, leaving 298 validation
  examples. This changes the evaluation population and can bias the result;
  it is not a score on the full holdout.
- Apple M5 / MPS, PyTorch 2.14.1; 2 layers, dimension 128, 4 heads, context
  128, batch 8, BPE vocab 512, learned from the first 20,000 training
  characters. Base: 500 updates, then SFT: 500 updates. One seed only.
- The pretraining text excluded the SFT validation examples. Its own
  train/validation loss is measured on a token split of the training text,
  not on the held-out conversations.

| SFT step | sampled train loss | sampled held-out loss |
|---:|---:|---:|
| 0 | 4.0103 | 4.1103 |
| 250 | 3.8400 | 3.8972 |
| 500 | 3.8041 | 3.8664 |

These are nats per assistant target token, sampled on different random batches
at each evaluation. They are not a chat-quality score, and this single run
does not isolate the effect of SFT from additional training.

The five handcrafted prompts cover a greeting, arithmetic, a simple
instruction, an umbrella question, and current weather. They are a qualitative
smoke test, not a benchmark or an independently verified uncontaminated test
set. Greedy decoding was used. The base receives `prompt + "\\n\\n"`; the chat
model receives the system/user/assistant template. These different contexts
make this a practical inspection, not a controlled causal comparison.

## Reproduce

Use fresh paths; checkpoints and downloaded data are git-ignored. This run's
weights remain local and are not included in the repository.

```bash
python -m data.prepare_chat --source alpaca --limit 3000 --shuffle --seed 0 \
  --out data/cache/phase-chat-train.jsonl --val-out data/cache/phase-chat-val.jsonl \
  --text-out data/cache/phase-chat-base.txt

python -m data.filter_chat_holdout --train data/cache/phase-chat-train.jsonl \
  --validation data/cache/phase-chat-val.jsonl --output data/cache/phase-chat-val-known.jsonl

python -m training.train --data data/cache/phase-chat-base.txt \
  --out checkpoints/phase-chat/base.pt --tokenizer bpe \
  --vocab-file checkpoints/phase-chat/bpe.json --vocab-size 512 \
  --bpe-max-chars 20000 --block-size 128 --n-embd 128 --n-head 4 --n-layer 2 \
  --batch-size 8 --max-steps 500 --warmup-steps 50 --eval-interval 250 \
  --eval-iters 10 --seed 0 --device mps --sample-tokens 0 --no-plot

python -m training.sft --data data/cache/phase-chat-train.jsonl \
  --val-data data/cache/phase-chat-val-known.jsonl --init-from checkpoints/phase-chat/base.pt \
  --out checkpoints/phase-chat/chat.pt --block-size 128 --n-embd 128 \
  --n-head 4 --n-layer 2 --batch-size 8 --max-steps 500 --warmup-steps 50 \
  --eval-interval 250 --eval-iters 10 --seed 0 --device mps --no-plot

python -m experiments.compare_chat --base checkpoints/phase-chat/base.pt \
  --chat checkpoints/phase-chat/chat-best.pt --device mps \
  --output checkpoints/phase-chat/comparison.json
```

The filter uses characters in the SFT training conversations, including system
messages. When using a separate base tokenizer, ensure it covers those
characters too; the SFT loader checks this. Filtering does not check
paraphrase leakage or data quality. Use a byte-level tokenizer in future work
to avoid excluding unfamiliar Unicode characters.
