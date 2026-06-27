# Results

First full run of both models on the same data and split.

## Setup

- Data: data/hinglish.csv, 5980 rows after normalization, balanced (2908 not-hate
  / 3074 hate). Source is HASOC 2019 Hindi (hindi_dataset.tsv train +
  hasoc2019_hi_test_gold_2919.tsv), pooled and deduped. Label map HOF -> 1,
  NOT -> 0.
- Split: stratified, seed 42, giving 4784 train / 598 val / 598 test. Both models
  used the identical split.
- Hyperparameters: max_len 128, batch_size 64, lr 2e-5, 4 epochs, full
  fine-tune, plain cross-entropy (no class weights, data is balanced).
- Hardware: single NVIDIA A10G (SageMaker ml.g5.xlarge).
- Best checkpoint chosen by validation macro-F1, scored once on the held-out
  test split.

## Test set comparison (598 examples)

| Model            | Test macro-F1 | Accuracy | F1 not_hate | F1 hate | Val macro-F1 |
|------------------|---------------|----------|-------------|---------|--------------|
| XLM-R (baseline) | 0.8445        | 0.8445   | 0.8463      | 0.8426  | 0.8344       |
| MuRIL            | 0.8528        | 0.8528   | 0.8528      | 0.8528  | 0.8411       |
| Delta            | +0.0083       | +0.0083  | +0.0065     | +0.0102 | +0.0067      |

Per-class test detail:

- XLM-R: not_hate P/R 0.8153 / 0.8797, hate P/R 0.8768 / 0.8111.
- MuRIL: not_hate P/R 0.8306 / 0.8763, hate P/R 0.8763 / 0.8306.

Confusion matrices (rows = true, cols = predicted):

```
XLM-R                  MuRIL
         not  hate              not  hate
not      256   35       not     255   36
hate      58  249       hate     52  255
```

## Read

MuRIL beats XLM-R on test macro-F1 by 0.0083 (0.83 points). The whole gain is on
the hate class: MuRIL's hate recall is 0.8306 vs XLM-R's 0.8111, so it recovers 6
more true-hate posts while giving up only 1 not-hate. MuRIL ends up perfectly
symmetric (both class F1 = 0.8528), while XLM-R leans slightly toward predicting
not-hate (lower hate recall, higher hate precision).

The margin is small, and that fits the data. HASOC 2019 Hindi is mostly native
Devanagari, not Roman-script Hinglish, so XLM-R is not strongly disadvantaged
here. The reason to expect a bigger MuRIL win is transliterated text, which this
dataset does not really contain. So this is a clean "MuRIL >= XLM-R" result, but
not yet a test of the transliteration thesis.

## What this does not yet cover

- Error analysis: read a sample of the 52 + 36 MuRIL mistakes for patterns
  (sarcasm, rare slang, label noise). Milestone 5.
- A run on genuinely code-mixed Roman-script Hinglish (HASOC 2021 ICHCL), which
  needs the conversation-tree JSON parser. That is where MuRIL should pull ahead
  more.
