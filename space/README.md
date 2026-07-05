---
title: Hinglish Hate Speech Detector
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Hinglish hate / offensive speech detector

A binary classifier for hate and offensive speech in Hinglish (Hindi-English
code-mixed text). Type a sentence and the model returns its call plus the
confidence split between the two classes.

## The model

This Space runs a fine-tuned [MuRIL](https://huggingface.co/google/muril-base-cased),
an India-specific multilingual encoder from Google pretrained on Indian languages
including transliterated text. It was fine-tuned on the HASOC 2019 Hindi dataset
(label 1 = hate/offensive, 0 = not), reaching a test macro-F1 of 0.853 on a
held-out split, slightly ahead of an XLM-RoBERTa baseline at 0.845.

## What it is good and bad at

The training data is mostly native Devanagari Hindi with some Roman-script
code-mixing, so the model handles both scripts. Two known weak spots, from the
error analysis: it tends to flag text that is profane but was labeled not-hate by
the annotators (label noise), and it misses implicit or sarcastic hate that has
no surface slurs. It never predicts with high confidence on the cases it gets
wrong, which is the signature of a model sitting at the data's ceiling.

This is a research demo, not a moderation product. Do not use it to make
decisions about real people.

## Provenance

Built from the hinglish-hate-detection project. MuRIL is Apache-2.0; the HASOC
data is released for research use.
