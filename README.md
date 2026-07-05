# Hinglish Hate Speech Detection

A text classifier for hate and offensive speech in Hinglish: Hindi-English
code-mixed social media text. Fine-tunes MuRIL and compares it against an
XLM-RoBERTa baseline. Reported metric is macro-F1, since the data is imbalanced.

## Why this is interesting

Most hate-speech detectors are trained on plain English. Real Indian social media
is not plain English. People switch between Hindi and English inside one sentence,
write Hindi words in Roman script (so "बकवास" becomes "bakwaas"), and add slang
and noise. That breaks English models and even trips up generic multilingual ones.

The question is whether India-specific pretraining (MuRIL, pretrained on Indian
languages including transliterated Hindi-English pairs) gives a measurable
advantage over a strong general multilingual model (XLM-RoBERTa). Same data,
same pipeline, same metric across all experiments.

## Results

### In-language fine-tuning (HASOC 2019 Hindi, 598-example test set)

Both models fine-tuned on the same Hindi training set and evaluated on the same
held-out split.

| Model            | Test macro-F1 | F1 not-hate | F1 hate | Val macro-F1 |
|------------------|---------------|-------------|---------|--------------|
| XLM-RoBERTa      | 0.845         | 0.846       | 0.843   | 0.834        |
| MuRIL            | 0.853         | 0.853       | 0.853   | 0.841        |
| Delta            | +0.008        | +0.007      | +0.010  | +0.007       |

MuRIL recovers 6 more true-hate posts, giving up 1 not-hate. It ends up
symmetric across both classes. The margin is small and fits the data: HASOC 2019
is mostly native Devanagari, not Roman-script Hinglish, so XLM-R is not strongly
disadvantaged on this corpus.

Error analysis on the 88 wrong MuRIL predictions: the most confident mistake sits
at p=0.74 (nothing wrong above 0.9), every error just clears the 0.5 boundary.
False positives are posts with surface profanity that annotators labeled not-hate;
false negatives are implicit or sarcastic political posts with no surface slurs.
Both failure modes are irreducible given the data.

### Cross-lingual zero-shot transfer (headline result)

Train on English hate speech (Davidson, ~25k tweets), evaluate zero-shot on the
same Hindi test split with no Hindi labels.

| Model            | Pretraining                        | Transfer macro-F1 | Hate recall |
|------------------|------------------------------------|-------------------|-------------|
| English BERT     | English only                       | 0.331             | 0.003       |
| mBERT            | 104 languages                      | 0.364             | 0.036       |
| XLM-RoBERTa      | 100 languages                      | 0.541             | 0.241       |
| MuRIL            | Indian languages + transliteration | 0.597             | 0.303       |

A monotonic climb across the pretraining-language gradient, 26.6 points floor to
ceiling. This is far outside the ~3-point confidence interval on 598 examples,
so it is a real effect.

English BERT cannot tokenize Devanagari; every post becomes near-identical unknown
tokens and it predicts one class. mBERT covers Hindi but barely moves from the
floor. What scales the transfer is the amount and relevance of Indian-language
pretraining: XLM-R (more Devanagari web text) reaches 0.54, MuRIL (India-specific
plus transliteration) reaches 0.60. MuRIL recovers 70% of the supervised ceiling
(0.597 / 0.853) with zero Hindi labels; XLM-R recovers 64%.

### Controlled script comparison (PRISM dataset)

Fine-tuned both models on each script subset of PRISM separately. If MuRIL's
transliteration pretraining matters, the advantage should be largest on Roman
Hinglish.

| Script                       | XLM-R | MuRIL | Delta  |
|------------------------------|-------|-------|--------|
| English (Latin)              | 0.844 | 0.812 | -0.032 |
| Hindi (Devanagari)           | 0.633 | 0.630 | -0.002 |
| Hinglish (Roman, code-mixed) | 0.755 | 0.734 | -0.021 |

The hypothesis is not supported. MuRIL does not outperform XLM-R on Roman
Hinglish. On Hindi and Hinglish the two are a statistical tie (confidence
intervals ~3-4 points on test sets of 478 to 1499 examples). XLM-R's modest
English edge is expected.

Methodology note: an initial run at 4 epochs showed MuRIL losing by 0.18 on
Hinglish. Per-epoch curves revealed the problem: MuRIL converged slower than
XLM-R on the small Roman subset (~60 steps per epoch). At 10 epochs both models
plateaued and the gap collapsed from 0.18 to 0.02. The lesson: compare at
convergence, not at a fixed epoch count.

## Main finding

When there is in-language supervision, full fine-tuning erases MuRIL's
pretraining advantage. The two models converge to the same performance across
all three in-language experiments, regardless of script. MuRIL's India-specific
pretraining gives a measurable edge only in the zero-shot regime, where the model
must rely on pretraining alone instead of task labels.

For a deployed Hinglish classifier with a labelled training set, XLM-RoBERTa is
a perfectly good choice. MuRIL earns its keep when you must transfer from another
language.

## Datasets

- HASOC 2019 Hindi (hasocfire.github.io): 5980 examples after normalization,
  2908 not-hate / 3074 hate. Label map HOF -> 1, NOT -> 0.
- PRISM Roman-Hinglish (Kaggle): English, Devanagari Hindi, and Roman code-mixed
  subsets under a consistent annotation scheme.
- Davidson English hate speech: ~25k tweets, used as the transfer source.

## Docs

- [docs/concepts.md](docs/concepts.md): the ML concepts the project rests on.
- [docs/design-decisions.md](docs/design-decisions.md): why MuRIL, why macro-F1,
  how class imbalance is handled.
- [docs/results.md](docs/results.md): full experiment writeups and error analysis.
