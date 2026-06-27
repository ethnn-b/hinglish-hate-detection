"""Model construction.

Thin wrapper around `AutoModelForSequenceClassification`. Given a preset name it
loads the matching pretrained encoder and attaches a fresh classification head
sized to `num_labels`. The head starts random and gets trained during
fine-tuning (see docs/concepts.md on sequence classification).
"""

from __future__ import annotations

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import PRESETS


def resolve_model_name(preset_or_name: str) -> str:
    """Accept either a preset key (`muril`) or a raw HF model id and return the id.

    Lets callers pass `muril` on the command line or a full path/id when loading
    a saved checkpoint.
    """
    return PRESETS.get(preset_or_name, preset_or_name)


def build_model(preset_or_name: str, num_labels: int = 2):
    """Load a pretrained encoder with a fresh classification head.

    `preset_or_name` is a preset key, a HF model id, or a local checkpoint path.
    """
    model_name = resolve_model_name(preset_or_name)
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )


def build_tokenizer(preset_or_name: str):
    """Load the tokenizer that matches the model.

    MuRIL uses WordPiece, XLM-R uses SentencePiece. AutoTokenizer picks the right
    one from the model id, so callers do not have to care which.
    """
    model_name = resolve_model_name(preset_or_name)
    return AutoTokenizer.from_pretrained(model_name)
