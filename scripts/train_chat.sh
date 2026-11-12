#!/usr/bin/env bash
# Train a chat checkpoint from public instruction data.
#
#   ./scripts/train_chat.sh                  # ~13M params, GPU recommended
#   PROFILE=tiny ./scripts/train_chat.sh     # ~3M params, CPU-friendly
#   DATA_LIMIT=3000 BASE_STEPS=500 SFT_STEPS=400 ./scripts/train_chat.sh
#
# Steps:
#   1. download Alpaca (or Dolly) and convert it to chat JSONL + plain text
#   2. pretrain a Base model on the same text
#   3. supervised fine-tune with assistant-only loss
#
# Outputs (CKPT_DIR defaults to checkpoints/chat):
#   base.pt, chat.pt, chat-best.pt, bpe.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
  else
    PYTHON=python
  fi
fi

PROFILE="${PROFILE:-small}"
case "$PROFILE" in
  tiny)
    BLOCK="${BLOCK:-128}"
    EMBD="${EMBD:-192}"
    HEAD="${HEAD:-4}"
    LAYER="${LAYER:-4}"
    VOCAB="${VOCAB:-2048}"
    BATCH="${BATCH:-8}"
    GRAD="${GRAD:-1}"
    BASE_STEPS="${BASE_STEPS:-600}"
    SFT_STEPS="${SFT_STEPS:-600}"
    ;;
  small)
    BLOCK="${BLOCK:-256}"
    EMBD="${EMBD:-384}"
    HEAD="${HEAD:-6}"
    LAYER="${LAYER:-6}"
    VOCAB="${VOCAB:-4096}"
    BATCH="${BATCH:-16}"
    GRAD="${GRAD:-2}"
    BASE_STEPS="${BASE_STEPS:-1500}"
    SFT_STEPS="${SFT_STEPS:-1200}"
    ;;
  medium)
    BLOCK="${BLOCK:-256}"
    EMBD="${EMBD:-512}"
    HEAD="${HEAD:-8}"
    LAYER="${LAYER:-8}"
    VOCAB="${VOCAB:-8192}"
    BATCH="${BATCH:-24}"
    GRAD="${GRAD:-2}"
    BASE_STEPS="${BASE_STEPS:-4000}"
    SFT_STEPS="${SFT_STEPS:-2500}"
    ;;
  *)
    echo "error: PROFILE must be tiny, small or medium (got '$PROFILE')" >&2
    exit 2
    ;;
esac

DATA_SOURCE="${DATA_SOURCE:-alpaca}"
DATA_LIMIT="${DATA_LIMIT:-8000}"
DATA_DIR="${DATA_DIR:-data/cache}"
CKPT_DIR="${CKPT_DIR:-checkpoints/chat}"
DEVICE="${DEVICE:-auto}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
BPE_MAX_CHARS="${BPE_MAX_CHARS:-500000}"
SEED="${SEED:-0}"

mkdir -p "$DATA_DIR" "$CKPT_DIR"

derive_warmup() {
  local warm=$(( $1 / 10 ))
  if (( warm > 100 )); then
    warm=100
  fi
  echo "$warm"
}

WARMUP_BASE="${WARMUP_BASE:-$(derive_warmup "$BASE_STEPS")}"
WARMUP_SFT="${WARMUP_SFT:-$(derive_warmup "$SFT_STEPS")}"

echo "==> profile: $PROFILE ($LAYER layers, $EMBD dim, context $BLOCK)"
echo "==> python:  $PYTHON"
echo "==> device:  $DEVICE"

echo
echo "==> 1/3 preparing $DATA_SOURCE data (limit $DATA_LIMIT)"
"$PYTHON" -m data.prepare_chat \
  --source "$DATA_SOURCE" \
  --limit "$DATA_LIMIT" \
  --shuffle --seed "$SEED" \
  --download-dir "$DATA_DIR" \
  --out "$DATA_DIR/chat.jsonl" \
  --text-out "$DATA_DIR/text.txt"

echo
echo "==> 2/3 pretraining Base model ($BASE_STEPS steps)"
"$PYTHON" -m training.train \
  --data "$DATA_DIR/text.txt" \
  --out "$CKPT_DIR/base.pt" \
  --tokenizer bpe \
  --vocab-file "$CKPT_DIR/bpe.json" \
  --vocab-size "$VOCAB" \
  --bpe-max-chars "$BPE_MAX_CHARS" \
  --block-size "$BLOCK" \
  --n-embd "$EMBD" \
  --n-head "$HEAD" \
  --n-layer "$LAYER" \
  --batch-size "$BATCH" \
  --grad-accum-steps "$GRAD" \
  --max-steps "$BASE_STEPS" \
  --warmup-steps "$WARMUP_BASE" \
  --eval-interval "$EVAL_INTERVAL" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --sample-tokens 0 \
  --no-plot

echo
echo "==> 3/3 chat SFT ($SFT_STEPS steps)"
"$PYTHON" -m training.sft \
  --data "$DATA_DIR/chat.jsonl" \
  --init-from "$CKPT_DIR/base.pt" \
  --out "$CKPT_DIR/chat.pt" \
  --block-size "$BLOCK" \
  --n-embd "$EMBD" \
  --n-head "$HEAD" \
  --n-layer "$LAYER" \
  --batch-size "$BATCH" \
  --grad-accum-steps "$GRAD" \
  --max-steps "$SFT_STEPS" \
  --warmup-steps "$WARMUP_SFT" \
  --eval-interval "$EVAL_INTERVAL" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --no-plot

echo
echo "==> done"
echo "terminal chat:"
echo "  $PYTHON -m inference.chat --ckpt $CKPT_DIR/chat-best.pt"
echo "web demo:"
echo "  $PYTHON -m server.app --ckpt $CKPT_DIR/chat-best.pt"
