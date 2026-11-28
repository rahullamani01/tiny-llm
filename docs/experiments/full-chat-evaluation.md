# Full-holdout chat evaluation

`python -m inference.evaluate_chat` computes deterministic teacher-forced
next-token loss over conversation JSONL. It scores every assistant content
token and closing `<|end|>` exactly once, ignores system/user/role targets, and
weights the result by the number of assistant target tokens. It never generates
answers and therefore does **not** measure whether replies are helpful, safe,
or correct.

```bash
python -m inference.evaluate_chat --ckpt checkpoints/phase-byte/chat-best.pt \
  --data data/cache/phase-chat-val.jsonl --device auto \
  --output checkpoints/phase-byte/full-eval.json
```

Use a trusted checkpoint and a genuinely held-out dataset. The optional
`--system` supplies a fallback system prompt when a conversation has none;
match it to the model's training setup. The report records the checkpoint and
dataset SHA-256, scoring method, device, PyTorch version, context length, loss,
perplexity, and counts. It will not overwrite an existing output file.

Each conversation is evaluated separately. Within it, inputs are divided into
non-overlapping chunks of at most `block_size` tokens; context is reset at every
chunk boundary. Thus a token near a chunk boundary may have much less context
than it would during generation. This method is deterministic but not a
sliding-window maximum-context score. It also differs from the randomly
sampled, batch-averaged training-time SFT loss; values should not be treated
as directly interchangeable. Perplexity is per tokenizer token, including end
tokens: do not compare it directly between byte and BPE tokenizers.

## First run

The local byte-tokenizer smoke checkpoint from
[byte-tokenizer.md](byte-tokenizer.md) was evaluated against the complete
300-conversation holdout on Apple M5 / MPS. It scored 79,404 assistant target
tokens, with loss **2.5877 nats/token** and perplexity **13.30**. The full
machine-readable report is [full-chat-eval-byte.json](full-chat-eval-byte.json).
The weights and dataset remain local and git-ignored; hashes identify the exact
inputs. This single metric does not establish an improvement in chat quality.
