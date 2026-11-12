# Chat SFT

This is the chat layer for tiny-llm: a template, a supervised fine-tuning (SFT)
loop, and a streaming terminal chat client. It is intentionally small enough to
read, but it is the same shape used by larger instruction-tuned models.

## What SFT changes

A Base model predicts the next token of whatever text it was trained on. SFT
teaches it to answer in a conversation format:

```text
<|system|>You are a helpful assistant.<|end|>
<|user|>Hi!<|end|>
<|assistant|>Hello! How can I help?<|end|>
```

Two things make this different from ordinary language-model training:

1. **Special tokens** mark roles and turn boundaries. They are appended after
   the normal character/BPE vocabulary, so old token ids never move.
2. **Loss masking**: the model reads the whole conversation, but only assistant
   content and the closing `<|end|>` contribute to the loss. Prompt tokens are
   labelled `-100`, which `torch.nn.functional.cross_entropy` ignores.

`chat/template.py` owns the format. `training/sft.py` owns data loading,
masking, batching and training. `inference/chat.py` owns the conversation loop.

## Train

```bash
# from scratch (good for testing the pipeline)
python -m training.sft --data data/chat_example.jsonl \
    --tokenizer bpe --vocab-size 2048 \
    --block-size 128 --n-embd 128 --n-head 4 --n-layer 4 \
    --max-steps 2000 --out checkpoints/tinygpt-chat.pt

# start from a Base checkpoint
python -m training.sft --data data/chat_example.jsonl \
    --init-from checkpoints/tinygpt-bpe.pt \
    --out checkpoints/tinygpt-chat.pt
```

`--init-from` keeps the Base architecture and copies its weights. Chat special
tokens increase the vocabulary, so the token embedding and LM head are resized:
old rows are copied, new rows are initialized, and the output head is re-tied if
the Base model used weight tying.

For a checkpoint trained on real instruction data, use
[`scripts/train_chat.sh`](../scripts/train_chat.sh) or the
[training recipe](train_chat.md).

Important flags:

| flag | meaning |
|---|---|
| `--data` | JSONL conversations |
| `--init-from` | optional Base checkpoint |
| `--resume` | continue an SFT run from `--out` |
| `--system` | default system prompt when a record has none |
| `--val-fraction` | conversation-level train/val split |
| `--block-size` | context window used for packed SFT windows |
| `--grad-accum-steps` | micro-batches per optimizer step |
| `--patience` | early stopping on validation plateau |

Checkpoints are written to `--out` (latest, resumable) and `chat-best.pt`
(best validation loss) next to it. The tokenizer state is stored inside the
checkpoint, including special tokens.

## Data format

One JSON object per line:

```json
{"system": "You are a helpful assistant.",
 "messages": [{"role": "user", "content": "Hi"},
              {"role": "assistant", "content": "Hello!"}]}
```

A bare list of messages is also accepted. Roles must be `system`, `user`, or
`assistant`; content must be non-empty.

`data/chat_example.jsonl` is a 10-conversation smoke test. Training on it will
not produce a useful assistant. For a toy chatbot, use something like:

- **TinyStories-Instruct** for simple, short, coherent instructions.
- **PersonaChat** or a small **Dolly** slice for multi-turn dialogue.
- Your own JSONL of short question/answer pairs.

A 10–30M parameter model with a 256–512 token context is a realistic target for
a CPU/small-GPU toy. Model quality depends far more on data and scale than on
the chat template.

## Chat

```bash
python -m inference.chat --ckpt checkpoints/tinygpt-chat.pt
python -m inference.chat --ckpt checkpoints/tinygpt-chat.pt --kv-cache
```

The CLI:

- keeps multi-turn history,
- trims old turns to fit `block_size`,
- streams one token at a time,
- stops when the model emits `<|end|>` or any other special token,
- supports `/system`, `/reset`, `/status`, `/temp`, `/topk`, `/topp`,
  `/penalty`, `/tokens`, `/help`, and `/quit`.

The model must be an SFT checkpoint. Loading a Base completion checkpoint gives
a clear error instead of silently producing role tokens.

## Web demo

```bash
pip install fastapi uvicorn
python -m server.app --ckpt checkpoints/tinygpt-chat.pt --port 8000
```

- `server/chat.py` validates request parameters, builds the chat prompt and
  yields streaming events. It has no FastAPI import, so it is tested directly.
- `server/app.py` exposes `POST /api/chat` (SSE), `GET /api/health`, serves the
  static UI at `/`, and serializes generation with a lock because the model is
  shared.
- `web/` is a dependency-free single-page client with multi-turn history, a
  stop/abort button, and a settings drawer for the system prompt, temperature,
  top-k, max tokens and KV caching.

The stream looks like:

```text
data: {"token": "Hel"}

data: {"token": "lo"}

data: {"done": true, "content": "Hello"}
```

Errors inside a stream are sent as `data: {"error": "..."}` so the client always
gets a well-formed response.

## Tests

```bash
python -m tests.test_chat
```

Covered: special-token round-trips for char and BPE vocabularies, prompt-masked
labels, context trimming, JSONL validation, SFT batch supervision, finite loss,
and vocabulary resizing when initializing from a Base checkpoint.

## Honest limitations

- The template has no tool calls, images, or function-calling format.
- There is no preference optimization (DPO/RLHF) after SFT.
- There is no safety classifier or content filter; this is a research toy.
- A tiny random or 10-conversation SFT run will not chat meaningfully. Treat the
  pipeline as the deliverable until you train on real data at a larger scale.
