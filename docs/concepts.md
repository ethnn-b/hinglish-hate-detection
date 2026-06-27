# Concepts

The ideas this project rests on, explained plainly. If you understand these, the
code is straightforward.

## Code-mixing and why it is hard

Code-mixing is switching between two languages inside a single utterance. A real
Hinglish example: "yaar this movie is total bakwaas, mat dekhna." That is English
and Hindi interleaved in one sentence. Three things make this hard for a model:

1. Script mixing. The same message can carry Devanagari ("बकवास") and Latin
   characters ("bakwaas") at once. A tokenizer that only really knows one script
   wastes vocabulary on the other.
2. Transliteration. Hindi gets written in Roman letters with no fixed spelling.
   "नहीं" shows up as "nahi", "nahin", "nai", "nhi". To a model these look like
   four different rare tokens unless it has seen transliterated Hindi in
   pretraining.
3. Sparse labeled data. There is a lot of labeled English hate speech and very
   little labeled Hinglish. So you cannot just train from scratch, you have to
   start from a pretrained model and fine-tune on a small set.

Put together, this is why a plain English classifier fails and why model choice
(does it know transliterated Hindi?) matters more than usual.

## Subword tokenization

Models do not see words, they see token IDs. The tokenizer splits text into
subword units so it can represent any string, including spellings it never saw,
by gluing pieces together. "bakwaas" might become `bak ##waas` or
`▁bak waas`. Two schemes show up here:

- WordPiece (used by BERT-family models, including MuRIL). Greedy longest-match
  against a fixed vocabulary, with `##` marking a continuation piece.
- SentencePiece / BPE (used by XLM-RoBERTa). Trains merges over raw bytes or
  characters, uses `▁` to mark a space. It does not need pre-tokenized words,
  which helps for languages without clean whitespace rules.

Why it matters here: code-mixed and transliterated words are rare, so they get
chopped into many small pieces. A tokenizer trained on Indian-language text
(MuRIL) keeps these as fewer, more meaningful pieces than one that was not. Fewer
pieces means the model spends less of its limited sequence length on one word and
keeps more signal.

## Transformer fine-tuning for sequence classification

The pretrained transformer already produces a contextual vector for every token.
For classification you do not need per-token outputs, you need one vector for the
whole sentence and a small layer on top.

- CLS pooling. BERT-style models prepend a special `[CLS]` token. Its final
  hidden state is treated as the sentence representation. (RoBERTa uses the
  equivalent `<s>` token at position 0.) You take that one vector.
- Classification head. A small linear layer (sometimes with a dropout and a
  tanh in between) maps that vector to `num_labels` logits. Here that is 2.
- Fine-tuning. You attach the head, then train the whole thing (transformer
  weights plus head) on your labeled data with a small learning rate. The
  pretrained weights already encode language; fine-tuning nudges them toward the
  task. `AutoModelForSequenceClassification` builds exactly this: base model plus
  a fresh head sized to `num_labels`.

The small learning rate (around 2e-5) matters. The pretrained weights are good
already, so you adjust them gently instead of overwriting what they learned.

## MuRIL pretraining

MuRIL (Multilingual Representations for Indian Languages) is a BERT-style model
from Google trained on 17 Indian languages plus English. The part that matters
for us: its pretraining data included transliterated text, meaning Romanized
Hindi paired with its native-script version. So the model has actually seen
"nahi" and "नहीं" and learned they are related. A generic multilingual model that
trained mostly on native-script web text has not had that exposure in the same
volume. That is the whole reason to expect MuRIL to win on Hinglish.

## Class imbalance

In hate-speech data the offensive class is usually the minority, often a small
fraction of all examples. If you do nothing, two problems follow:

- The loss is dominated by the majority class, so the model can score high
  accuracy by mostly predicting "not hate" and barely learning the minority
  class.
- Naive accuracy hides this completely.

Standard fixes (covered in design-decisions.md): weight the loss so minority
errors cost more (class weights), or use focal loss to down-weight easy examples.
Either way you must measure with a metric that does not let the majority class
mask poor minority performance.

## Macro-F1 vs accuracy

Accuracy is (correct / total). On imbalanced data it is misleading: predict the
majority class every time and accuracy looks fine while the model is useless.

F1 is the harmonic mean of precision and recall for one class, so it punishes a
model that ignores that class. Macro-F1 computes F1 separately for each class and
then averages the per-class scores with equal weight. Because the rare class
counts as much as the common one, you cannot game macro-F1 by predicting the
majority. That is why it is the headline metric here. (Micro-F1, by contrast,
pools all decisions and ends up close to accuracy on this kind of data, so it is
not what we report.)

## Evaluation hygiene

A few rules so the numbers mean something:

- Split first, normalize and tokenize after, and fit nothing (no class weights,
  no thresholds) on the test set. The test set is touched once, at the end.
- Use a stratified split so train, val, and test all have the same class ratio.
  Otherwise a lucky or unlucky split moves the score for the wrong reason.
- Pick the best checkpoint on validation macro-F1, not on test. Test is only for
  the final reported number.
- Compare MuRIL and XLM-RoBERTa on the exact same test split with the exact same
  preprocessing, or the comparison is meaningless.
- Watch per-class recall, not just the averaged number. A good macro-F1 with
  terrible minority recall is a sign something is off.
