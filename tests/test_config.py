"""Tests for Config preset resolution and output paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from hatedetect.config import Config


def test_muril_preset_resolves():
    assert Config(preset="muril").model_name == "google/muril-base-cased"


def test_xlmr_preset_resolves():
    assert Config(preset="xlmr").model_name == "xlm-roberta-base"


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        _ = Config(preset="bert").model_name


def test_output_dir_follows_preset():
    cfg = Config(preset="muril", models_dir=Path("models"))
    assert cfg.output_dir == Path("models/muril-best")


def test_defaults_match_spec():
    cfg = Config()
    assert cfg.max_len == 128
    assert cfg.batch_size == 32
    assert cfg.lr == 2e-5
    assert cfg.epochs == 4
    assert cfg.num_labels == 2
