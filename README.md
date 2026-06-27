# Hinglish Hate Speech Detection

A text classifier for hate and offensive speech in Hinglish: Hindi-English
code-mixed social media text. It fine-tunes MuRIL (an India-focused multilingual
transformer) and compares it against an XLM-RoBERTa baseline. The reported metric
is macro-F1, since the data is imbalanced.

## Why this is interesting

Most hate-speech detectors are trained on plain English. Real Indian social media
is not plain English. People switch between Hindi and English inside one
sentence, write Hindi words in Roman script (so "बकवास" becomes "bakwaas"), and
add slang and noise. That breaks English models and even trips up generic
multilingual ones.

The angle here is a head-to-head: an India-specific model (MuRIL, pretrained on
Indian languages including transliterated Hindi-English pairs) against a strong
general multilingual model (XLM-RoBERTa). Same data, same pipeline, same metric.
The project also takes text normalization and class imbalance seriously instead
of throwing raw text at a model and hoping.

## Features

- Clean `text,label` data pipeline with a pure, unit-tested normalization step.
- Stratified train/val/test split so the class ratio is preserved across splits.
- Two interchangeable model presets (MuRIL, XLM-RoBERTa) behind one config.
- Full fine-tune with AdamW and linear warmup, best checkpoint chosen by
  validation macro-F1.
- Evaluation that reports macro-F1, per-class precision/recall/F1, and a
  confusion matrix.
- A small Gradio demo for trying single inputs.

## Setup

Uses [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv venv
uv sync
```

## Usage

```bash
# convert a raw dataset download into the training CSV
uv run python scripts/get_data.py --raw data/hasoc_raw.csv --out data/hinglish.csv

# train the baseline
uv run python -m hatedetect.train --preset xlmr --data data/hinglish.csv

# train MuRIL
uv run python -m hatedetect.train --preset muril --data data/hinglish.csv

# evaluate a checkpoint
uv run python -m hatedetect.evaluate --model models/muril-best --data data/hinglish.csv

# predict on a single string
uv run python -m hatedetect.infer --model models/muril-best --text "kya bakwaas hai yaar"

# launch the demo
uv run python -m hatedetect.app
```

## Folder structure

```
hinglish-hate-detection/
  CLAUDE.md              project context and module spec
  README.md             this file
  pyproject.toml
  docs/
    concepts.md         the ML concepts behind the project
    design-decisions.md why MuRIL, why macro-F1, etc.
  src/hatedetect/       the package (config, data, model, train, evaluate, infer, app)
  tests/                unit tests
  scripts/              data prep helpers
  data/                 CSVs and raw downloads (gitignored)
  models/              saved checkpoints (gitignored)
```

## Datasets

These are public research datasets. Download them yourself, then run
`scripts/get_data.py` to convert into the `text,label` CSV the pipeline expects.

- HASOC Hindi-English code-mixed: https://hasocfire.github.io/
- Dravidian-CodeMix: https://dravidianlangtech.github.io/
- GLUECoS (alternative code-switching benchmark): https://microsoft.github.io/GLUECoS/

## Results

Not run yet. This section will hold:

| Model         | Test macro-F1 | F1 (not-hate) | F1 (hate) |
| ------------- | ------------- | ------------- | --------- |
| XLM-RoBERTa   | TBD           | TBD           | TBD       |
| MuRIL         | TBD           | TBD           | TBD       |

Plus the confusion matrix for the best model and a short error analysis of the
cases it gets wrong.
