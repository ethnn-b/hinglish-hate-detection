# Design decisions

Each choice with the alternatives considered and why this one. Where a decision
is reversible, that is noted.

## Model: MuRIL vs IndicBERT vs XLM-RoBERTa

Decision: fine-tune MuRIL as the main model, use XLM-RoBERTa as the baseline.

- MuRIL (`google/muril-base-cased`). Pretrained on 17 Indian languages plus
  English, and importantly on transliterated pairs (Romanized Hindi mapped to
  Devanagari). That directly matches Hinglish.
  - Pros: built for exactly this text, has seen transliteration, base size fits
    on modest hardware.
  - Cons: less famous than XLM-R, fewer community checkpoints to compare against.
- IndicBERT. Also Indian-language focused, lighter (ALBERT-style parameter
  sharing).
  - Pros: smaller and faster, good Indian-language coverage.
  - Cons: parameter sharing can cap accuracy on a hard task, and reported results
    on code-mixed hate speech are mixed. Kept as a possible v2 third contender,
    not in v1.
- XLM-RoBERTa (`xlm-roberta-base`). Strong general multilingual model.
  - Pros: very strong baseline, widely used, easy to justify as the thing to
    beat.
  - Cons: trained mostly on native-script web text, so less direct exposure to
    Romanized Hindi.

Using XLM-R as the baseline makes the result a clean story: does an
India-specific model actually beat a strong general one on this data.

## Tuning: full fine-tune vs LoRA

Decision: full fine-tune in v1.

- Full fine-tune updates all weights.
  - Pros: best accuracy for base-size models, simplest code path, no extra
    library.
  - Cons: more memory and slower, one full copy of weights per saved checkpoint.
- LoRA (low-rank adapters) freezes the base and trains small adapter matrices.
  - Pros: much smaller memory and tiny checkpoints, easy to keep many task
    variants.
  - Cons: extra dependency and config, and on base-size models the headroom over
    a full fine-tune is small. The savings matter most on large models, which
    these are not.

Base-size encoders fit fine, so the simpler full fine-tune wins for v1. LoRA is a
reasonable add later if checkpoint size or training many variants becomes a pain.

## Preprocessing: normalization vs raw text

Decision: light normalization, kept as a pure function.

- Raw text. Feed strings straight to the tokenizer.
  - Pros: zero preprocessing bugs, the model sees exactly what was posted.
  - Cons: URLs and @mentions are noise that eat sequence length and carry no
    signal about whether something is hateful.
- Light normalization (strip URLs, strip @mentions, collapse repeated
  whitespace).
  - Pros: removes obvious noise, cheap, easy to unit test.
  - Cons: a tiny risk of removing something that mattered (rare).
- Heavy normalization (lowercasing, stripping all punctuation, transliterating
  everything to one script).
  - Cons: destroys signal. Casing and repeated characters ("loooser",
    "STUPID") carry intent, and forcing one script throws away exactly the
    code-mixing the model is supposed to handle. Avoided.

The middle option is the choice. `normalize()` stays pure (string in, string out)
so it is trivially testable and the same function runs at train and inference
time.

## Dataset choice

Decision: HASOC Hindi-English code-mixed as primary, Dravidian-CodeMix as a
second source, GLUECoS as a fallback benchmark.

- HASOC. A recognized shared task with Hindi-English code-mixed hate/offensive
  labels.
  - Pros: established, citable, fits the binary hate/not-hate framing well.
  - Cons: tweet-era data, some noise and label disagreement.
- Dravidian-CodeMix. Code-mixed offensive language for Dravidian languages.
  - Pros: more code-mixed data, useful for a second source or v2.
  - Cons: different language pairs (Tamil, Malayalam, Kannada with English), so
    not directly the same task as Hindi-English.
- GLUECoS. A broader code-switching benchmark.
  - Pros: standardized splits, good for sanity-checking the pipeline.
  - Cons: covers several tasks, not purely hate speech.

Start with HASOC because it is the cleanest match. Document the source-label to
binary mapping wherever the conversion happens.

## Metric: macro-F1

Decision: report macro-F1 as the headline, with per-class F1 and a confusion
matrix alongside.

- Accuracy. Rejected as the headline: on imbalanced data it rewards predicting
  the majority.
- Micro-F1. Pools all decisions, so on this data it sits close to accuracy and
  hides minority performance.
- Macro-F1. Averages per-class F1 with equal weight, so the rare class counts as
  much as the common one. This is what we want to optimize and report.
- Weighted-F1. Weights each class by its support, which drifts back toward the
  majority. Not the headline, fine to log as extra context.

Per-class numbers and the confusion matrix are always shown too, because a single
averaged number can hide a class the model is failing on.

## Imbalance handling: class weights vs focal loss

Decision: start plain, add class-weighted loss if minority recall is poor, keep
focal loss as a fallback.

- Plain cross-entropy. Simplest, sometimes enough if imbalance is mild.
- Class weights. Scale each class's loss by the inverse of its frequency so
  minority mistakes cost more.
  - Pros: one line in the loss, well understood, usually the right first move.
  - Cons: can hurt precision if you over-weight, needs a quick tune.
- Focal loss. Down-weights easy, already-correct examples so training focuses on
  the hard ones.
  - Pros: strong when the minority is both rare and hard.
  - Cons: an extra hyperparameter (gamma) and more code. Reach for it only if
    class weights are not enough.

This ordering keeps v1 simple and only adds complexity when the validation
numbers say it is needed.

## Max sequence length

Decision: `max_len = 128`.

Code-mixed social posts are short, and tokenizing transliterated words into many
subwords inflates the token count, so 128 covers the large majority of examples
with headroom. Cost grows with sequence length, so going to 256 or 512 would
slow training for almost no extra coverage. If a length histogram of the actual
data shows heavy truncation, bump it. Cheap to change.

## Split strategy

Decision: stratified train/val/test, fixed seed.

- Random split. Risk: class ratio drifts between splits, especially with a small
  minority class, which adds noise to the comparison.
- Stratified split. Preserves the class ratio in every split, so train, val, and
  test all look like the real distribution. A fixed seed makes runs reproducible
  and keeps the MuRIL and XLM-R comparison on identical splits.

Val is for picking the best checkpoint by macro-F1. Test is held out and scored
once at the end.
