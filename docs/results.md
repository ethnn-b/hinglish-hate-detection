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

## Error analysis (in-language MuRIL)

Ran scripts/error_analysis.py on the in-language MuRIL checkpoint over the 598
test examples (88 wrong: 36 false positives, 52 false negatives). Two clear
patterns, and one striking fact about confidence.

Confidence: the most confident mistake in the whole test set is p=0.74. Nothing
is wrong at 0.9. Every error sits just over the 0.5 boundary, so the model is
hedging on genuinely ambiguous examples, not blundering. That is the signature of
a model at the data's ceiling.

False positives (true not-hate, predicted hate): almost all are loaded with slurs
and profanity (harami, randi, kamine, dalle, kutte, neech) but labeled not-hate
in the gold data. The model learned the reasonable rule "explicit profanity ->
offensive" and fired; the annotators disagreed. This is label noise, and it is
unwinnable for any model.

False negatives (true hate, predicted not-hate): implicit or sarcastic political
posts with no surface profanity. Catching them needs world knowledge and context,
not transliteration. A few are Roman-script Hinglish and one is Marathi, not even
Hindi.

Takeaway: the error budget is dominated by label noise and context-dependent
hate, neither of which MuRIL's transliteration edge can fix, and the data is
native Devanagari where XLM-R is already competent. That is why the in-language
gap is within noise. It motivated the pivot below.

## Cross-lingual zero-shot transfer (headline result)

Setup: train each model on English hate speech (Davidson, ~25k tweets, hate or
offensive -> 1, neither -> 0) and evaluate zero-shot on the same Hindi test split
as above. No model sees a single Hindi label. Same hyperparameters (batch 64,
4 epochs, seed 42). Models span a pretraining-language gradient.

| model  | pretraining                        | transfer macro-F1 | hate recall |
|--------|------------------------------------|-------------------|-------------|
| enbert | English only                       | 0.3309            | 0.003       |
| mbert  | 104 languages                      | 0.3643            | 0.036       |
| xlmr   | 100 languages                      | 0.5409            | 0.241       |
| muril  | Indian languages + transliteration | 0.5967            | 0.303       |

The result is a monotonic climb along the gradient with a 26.6-point spread from
floor to MuRIL. Unlike the in-language 0.8-point gap, this is far larger than the
~3-point confidence interval on a 598-example test set, so it is a real effect.

Read:

- "Multilingual" is not enough. mBERT covers Hindi but barely clears the floor
  (0.36). What scales the transfer is the amount and relevance of Indian-language
  pretraining: XLM-R (more Devanagari web text) reaches 0.54, MuRIL (India
  specific) tops it at 0.60.
- The mechanism is hate recall. Every English-trained model is conservative on
  Hindi and leans toward not-hate, but the share of actual hate it still catches
  scales with pretraining: enbert/mbert near zero, XLM-R 0.24, MuRIL 0.30. Hate
  cues are lexical and cultural and do not cross languages easily; MuRIL crosses
  that gap best.
- Recovery of the supervised ceiling. Against the in-language fine-tuned numbers
  (~0.85), MuRIL recovers 70% of the ceiling (0.597 / 0.853) with zero Hindi
  labels, vs XLM-R's 64%. enbert collapses (it cannot tokenize Devanagari, so
  every post becomes near-identical unknown tokens and it predicts one class).
- MuRIL scored 0.94 macro-F1 on the English validation set, then 0.60 zero-shot
  on Hindi. That contrast isolates the cross-lingual transfer gap itself.

Caveats: single seed per model, so the small mbert-vs-enbert and
xlmr-vs-muril gaps individually are not significance-tested, but the overall
gradient and the floor-to-MuRIL spread are far outside noise. The English data is
imbalanced (~80% positive); it is identical across all four models, so the
comparison is fair, but absolute hate recall would shift with class weighting.

## Controlled script comparison (PRISM): does MuRIL's edge depend on script?

Setup: PRISM labels each row english / hindi (Devanagari) / hinglish (Roman
code-mixed) under one annotation scheme. We fine-tuned MuRIL and XLM-R
in-language on each subset and scored them on that subset's test split. Holding
the dataset and labels constant, only the script changes. Hypothesis: MuRIL's
advantage would be largest on Roman Hinglish (its transliteration home turf).

Result (10 epochs, batch 32, best checkpoint by validation macro-F1):

| script                       | XLM-R  | MuRIL  | MuRIL - XLM-R |
|------------------------------|--------|--------|---------------|
| English (Latin)              | 0.8440 | 0.8118 | -0.032        |
| Hindi (Devanagari)           | 0.6326 | 0.6303 | -0.002        |
| Hinglish (Roman, code-mixed) | 0.7545 | 0.7335 | -0.021        |

The hypothesis is not supported. The gap is not largest on Hinglish; it is small
everywhere and slightly favors XLM-R. On Hindi and Hinglish the two are a
statistical tie: on the Hinglish validation set MuRIL actually beat XLM-R (0.725
vs 0.701) and then lost on test (0.734 vs 0.755). A ranking that flips between
validation and test is the signature of a difference within noise (test sets are
478 to 1499 examples, so confidence intervals are roughly +/- 3 to 4 points).
XLM-R is modestly better on English, which is unsurprising.

Methodology note (a real catch). A first run at 4 epochs showed MuRIL losing by
0.18 on Hinglish, which looked like a strong negative result. The per-epoch
curves showed why it was wrong: MuRIL sat at the predict-majority floor for two
epochs and was still climbing at epoch 4, while XLM-R had already converged. On
the small Roman subset (3818 train rows, about 60 steps per epoch at batch 64),
4 epochs was not enough for the slower-converging model. At 10 epochs both models
plateaued (MuRIL Hinglish peaked at epoch 6, XLM-R at epoch 7), and the gap
collapsed from 0.18 to 0.02. The lesson: compare models at convergence, not at a
fixed epoch budget. PRISM is also a Kaggle compilation, so treat absolute numbers
with caution; the within-dataset comparison is what matters.

## Synthesis across all three experiments

Three setups now point the same way:

1. In-language HASOC Hindi (Devanagari): MuRIL 0.853 vs XLM-R 0.845. Tie.
2. In-language PRISM (English, Devanagari, Roman Hinglish): tie on Hindi and
   Hinglish, XLM-R slightly ahead on English.
3. Zero-shot cross-lingual transfer (train English, test Hindi): MuRIL 0.597 vs
   XLM-R 0.541, and a clean monotonic gradient from English-only BERT (0.33) up
   to MuRIL.

The pattern: when there is in-language supervision, full fine-tuning erases
MuRIL's pretraining advantage and the two models converge to the same
performance, regardless of script, including Roman Hinglish. MuRIL's
India-specific pretraining gives a measurable edge only in the zero-shot regime,
where the model has no target-language labels and must rely on pretraining alone.

That is the project's real finding, and it is more precise than "MuRIL wins":
pretraining choice matters most when task supervision is scarce, and washes out
once you fine-tune on enough in-language data. It also means that for a deployed
Hinglish classifier with a labelled training set, XLM-R is a perfectly good
choice; MuRIL earns its keep when you must transfer from another language.

## What this does not yet cover

- Phase 2: a local open LLM (Llama-3.1-8B / Qwen2.5-7B on the A10G) as a zero-shot
  baseline against these fine-tuned models, and an LLM label-noise audit to put a
  number on the HASOC noise found in the error analysis.
- Multiple seeds per model to error-bar the transfer table.
- A run on genuinely code-mixed Roman-script Hinglish (HASOC 2021 ICHCL), where
  MuRIL's transliteration pretraining should help even more.
