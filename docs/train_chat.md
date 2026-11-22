# Training a real chat checkpoint

`data/chat_example.jsonl` has 10 conversations. It proves the SFT plumbing, but
it cannot teach a model to answer questions. This page is the recipe for a
checkpoint that can at least produce short, on-topic replies: real instruction
data, a small Base pretraining run, then SFT.

The honest ceiling: a 3–35M parameter model trained for minutes to a few hours
will still be repetitive and factually unreliable. The deliverable is the full
pipeline — data conversion, pretraining, masked SFT, and serving — not ChatGPT.

## One command

```bash
./scripts/train_chat.sh                  # small profile, ~14M params
PROFILE=tiny ./scripts/train_chat.sh     # ~2–3M params, CPU-friendly
```

The script runs three stages:

1. **Prepare data** — download Alpaca (52k instructions), group matching user
   prompts and hold out `data/cache/chat-val.jsonl`; only training conversations
   go into `data/cache/chat.jsonl` and the Base `data/cache/text.txt`.
2. **Pretrain Base** — `training.train` on the plain text, BPE tokenizer,
   `checkpoints/chat/base.pt`.
3. **SFT** — `training.sft --init-from base.pt`, assistant-only loss,
   `checkpoints/chat/chat.pt` and `chat-best.pt`.

Then:

```bash
python -m inference.chat --ckpt checkpoints/chat/chat-best.pt
python -m server.app --ckpt checkpoints/chat/chat-best.pt
```

## Profiles

| `PROFILE` | layers | dim | heads | context | vocab | params | intended hardware |
|---|---|---|---|---|---|---|---|
| `tiny` | 4 | 192 | 4 | 128 | 2048 | ~2–3M | CPU, quick test |
| `small` | 6 | 384 | 6 | 256 | 4096 | ~14M | one GPU (T4 or better) |
| `medium` | 8 | 512 | 8 | 256 | 8192 | ~34M | a larger GPU |

Defaults for `small`: batch 16 × grad-accum 2, 1500 Base steps, 1200 SFT steps.
On a free Colab T4 expect roughly tens of minutes, depending on throttling. CPU
runs of `small` are possible but slow; use `tiny` when you only want to verify
the pipeline.

Every setting can be overridden by an environment variable:

```bash
DATA_SOURCE=dolly DATA_LIMIT=15000 PROFILE=small \
BASE_STEPS=2500 SFT_STEPS=2000 DEVICE=cuda \
./scripts/train_chat.sh
```

| variable | default | meaning |
|---|---|---|
| `PROFILE` | `small` | `tiny`, `small`, or `medium` |
| `DATA_SOURCE` | `alpaca` | `alpaca`, `dolly`, or `input` |
| `DATA_LIMIT` | `8000` | conversations to keep (0 = all) |
| `DEVICE` | `auto` | `auto`, `cpu`, `mps`, `cuda` |
| `BASE_STEPS` | profile | Base pretraining steps |
| `SFT_STEPS` | profile | SFT steps |
| `BATCH` / `GRAD` | profile | batch size and gradient accumulation |
| `DATA_DIR` | `data/cache` | prepared data (git-ignored) |
| `CKPT_DIR` | `checkpoints/chat` | checkpoints and BPE vocabulary |
| `EVAL_INTERVAL` | `200` | evaluation/logging interval |
| `SEED` | `0` | seed for data shuffling and training |

## Data

`data/prepare_chat.py` accepts both common instruction shapes:

- **Alpaca** — `instruction`, optional `input`, `output`.
- **Dolly** — `instruction`, optional `context`, `response`.

```bash
# download and keep 8k shuffled examples
python -m data.prepare_chat --source alpaca --limit 8000 --shuffle \
    --out data/cache/chat.jsonl --text-out data/cache/text.txt

# Dolly
python -m data.prepare_chat --source dolly --limit 15000 \
    --out data/cache/dolly.jsonl --text-out data/cache/dolly.txt

# your own file, in either shape
python -m data.prepare_chat --source input --input my_data.jsonl --out data/cache/mine.jsonl
```

Flags: `--max-chars` drops examples that would not fit usefully in the context
window (default 2000), `--format text` writes plain text instead of chat JSONL,
and `--shuffle --seed` makes the subset reproducible.

The converter deduplicates identical instruction/answer pairs and skips empty
records. For a held-out split before pretraining, add
`--val-out data/cache/chat-val.jsonl`. The `--out` and `--text-out` files then
contain only training conversations. Pass `--val-data data/cache/chat-val.jsonl`
to SFT. Prompt groups are matched after case-folding and whitespace
normalization; similar paraphrases are not detected. Splitting by group means
the actual conversation fraction can differ from `--val-fraction`.
Downloads are cached in `data/cache/`, which is git-ignored.

The base model and any externally supplied tokenizer must also be trained
without the held-out data for the SFT validation loss to be a genuine holdout.
The recipe does this, but cannot verify an externally supplied checkpoint's
training data. The recipe requires a fresh `CKPT_DIR` so it cannot silently reuse
an old vocabulary or overwrite checkpoints. Validation characters absent from the training vocabulary cause
a clear error; use more representative training data rather than learning a
vocabulary from the holdout.

Alpaca is a research dataset generated with OpenAI models; Dolly is released by
Databricks. Check each dataset's license before redistributing derived data.

## Using your own data

The best small-model results usually come from a narrow domain. Write JSONL in
the SFT format:

```json
{"system": "You are a support assistant for a bike shop.",
 "messages": [
   {"role": "user", "content": "Do you ship to Canada?"},
   {"role": "assistant", "content": "Yes, shipping to Canada takes 5–8 days."}
 ]}
```

A few thousand consistent examples beat 50k unrelated ones. Keep answers short
(one to three sentences), include the same system prompt you will serve with,
and hold out a validation split (SFT does this automatically per conversation).
For multi-stage runs, split before Base pretraining as above, rather than
relying on SFT's automatic split afterwards.

## Pipeline details

**Why pretrain before SFT?** SFT teaches format and behavior; it does not teach
English from scratch efficiently. A short Base run on the instruction text gives
the model basic fluency, then SFT shapes it into question/answer turns. With very
little data you can skip Base training, but quality drops.

**Vocabulary.** Base training learns a BPE vocabulary at `CKPT_DIR/bpe.json`.
SFT loads the Base checkpoint's tokenizer state and appends chat special tokens;
old ids never move, and the embedding/LM head are resized automatically.

**Loss.** Only assistant content and the closing `<|end|>` token are trained.
System prompts, user turns, and role markers are masked with `-100`.
Inputs predict the next token, with labels shifted by one position. Earlier
versions mistakenly trained on the current token. Old SFT checkpoints may have
learned that incorrect objective; start a new SFT run rather than resuming one.

**Checkpoints.** `base.pt` is the latest Base checkpoint, `chat.pt` the latest
resumable SFT checkpoint, and `chat-best.pt` the best validation loss. Resume an
interrupted SFT run with:

```bash
python -m training.sft --data data/cache/chat.jsonl --out checkpoints/chat/chat.pt \
    --block-size 256 --n-embd 384 --n-head 6 --n-layer 6 --resume
```

## Colab

Open `notebooks/train_chat_colab.ipynb` in Google Colab, choose **Runtime →
Change runtime type → GPU**, and run the cells. It downloads the repo, trains
with the `small` profile, prints a sample reply, launches the web demo through
Colab's port proxy, and offers the checkpoint for download.

## What to expect

A successful run produces short answers in the right format. Expect:

- repetitive phrasing and occasional loops,
- confident wrong facts,
- weak multi-turn memory,
- no reasoning, tool use, or safety behavior.

To go further: train longer, add more diverse instruction data, raise the
context window, and only then consider preference optimization. Scale and data
quality dominate; the chat template is the easy part.
