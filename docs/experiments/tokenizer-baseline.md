# First tokenizer baseline

## Question

How do character and BPE tokenization affect sequence length and early model
behavior on tiny Shakespeare? This is a short pipeline baseline, not a
conclusive model-quality or speed benchmark.

## Reproduce

From the repository root, install the dependencies and download the dataset
using the README setup, then run:

```bash
python -m experiments.tokenizer_baseline --device auto --output checkpoints/my-baseline.json
```

The output must be a new filename; the script won't overwrite an existing
result. It records dataset SHA-256, configuration, environment, token counts,
model sizes, loss, timing, and fixed-prompt greedy samples. It does not save
model checkpoints. Use `--help` to change the settings.

The script splits the raw text chronologically, then builds each tokenizer from
the training portion only. BPE merges are learned from its first 20,000
characters. It fails clearly if the validation text contains a character absent
from training, rather than learning the validation vocabulary. Both models
use the same raw text split, seed and hyperparameters; they train for the same
number of token batches. Training text exposure in characters is therefore
different.

## First run

The full output is in [tokenizer-baseline-m5.json](tokenizer-baseline-m5.json).
It used Apple M5 / MPS, PyTorch 2.14.1, seed 1337, 100 steps, batch size 8,
context 64, 2 layers, 64 embedding dimensions, and 4 heads. The validation
split is the last 10% of the raw text. These results are from a single run:

| Measurement | Character | BPE |
|---|---:|---:|
| Vocabulary | 65 | 256 |
| Parameters | 112,193 | 136,832 |
| Validation tokens | 111,540 | 62,881 |
| Validation characters per token | 1.00 | 1.77 |
| Validation loss (nats/token) | 2.633 | 4.289 |

BPE represented the same validation text with about 44% fewer tokens. The
sampled loss is measured per **different kind of token**, so the two loss values
are not directly comparable and do not establish a winner. Both models produced
repetitive greedy samples after only 100 steps.

The saved timings are rough observations only: tests ran concurrently, and
device warmup, tokenizer preparation, model size and text exposure differ.
Neither runtime nor the two samples supports a conclusion about speed or
quality. Future work should evaluate both on a common unit such as bits per
byte, use multiple seeds, and compare matched compute or data budgets. The
[longer comparison](tokenizer-long.md) takes the first two steps.
