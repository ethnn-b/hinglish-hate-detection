"""Tests for the data pipeline.

`normalize()` is the first thing to test (CLAUDE.md): it is pure, so no model
weights or IO are needed. We also test the stratified split keeps class ratios
and the Dataset tokenizes with an injected fake tokenizer.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from hatedetect.data import HateDataset, normalize, split_data


class TestNormalize:
    def test_strips_url_http(self):
        assert normalize("check this https://example.com/x now") == "check this now"

    def test_strips_url_www(self):
        assert normalize("see www.example.com please") == "see please"

    def test_strips_mention(self):
        assert normalize("@someuser kya bakwaas hai") == "kya bakwaas hai"

    def test_collapses_whitespace(self):
        assert normalize("too    many\t\nspaces") == "too many spaces"

    def test_strips_leading_trailing(self):
        assert normalize("   hello   ") == "hello"

    def test_keeps_casing_and_repeats(self):
        # Casing and repeated chars carry intent, so they must survive.
        assert normalize("STUPID loooser") == "STUPID loooser"

    def test_keeps_devanagari(self):
        # Code-mixing is the point; both scripts must survive.
        assert normalize("yeh बकवास hai") == "yeh बकवास hai"

    def test_url_and_mention_together(self):
        assert normalize("@bob look https://x.io haha") == "look haha"

    def test_empty_string(self):
        assert normalize("") == ""

    def test_only_noise_becomes_empty(self):
        assert normalize("@bob https://x.io") == ""

    def test_is_pure_no_mutation(self):
        # Same input, same output, called twice.
        s = "  @a https://b.c  hi  "
        assert normalize(s) == normalize(s) == "hi"


class TestSplit:
    def _frame(self, n_pos=20, n_neg=80):
        texts = [f"pos {i}" for i in range(n_pos)] + [f"neg {i}" for i in range(n_neg)]
        labels = [1] * n_pos + [0] * n_neg
        return pd.DataFrame({"text": texts, "label": labels})

    def test_split_sizes_sum_to_total(self):
        df = self._frame()
        train, val, test = split_data(df, val_frac=0.1, test_frac=0.1, seed=42)
        assert len(train) + len(val) + len(test) == len(df)

    def test_fractions_are_of_total(self):
        df = self._frame()  # 100 rows
        train, val, test = split_data(df, val_frac=0.1, test_frac=0.1, seed=42)
        assert len(test) == 10
        assert len(val) == 10
        assert len(train) == 80

    def test_stratified_keeps_ratio(self):
        df = self._frame(n_pos=20, n_neg=80)  # 20% positive
        train, val, test = split_data(df, val_frac=0.1, test_frac=0.1, seed=42)
        for split in (train, val, test):
            frac_pos = split["label"].mean()
            assert abs(frac_pos - 0.2) < 0.05

    def test_deterministic_with_seed(self):
        df = self._frame()
        a = split_data(df, seed=7)
        b = split_data(df, seed=7)
        for x, y in zip(a, b):
            assert x["text"].tolist() == y["text"].tolist()

    def test_no_overlap_between_splits(self):
        df = self._frame()
        train, val, test = split_data(df, seed=42)
        sets = [set(s["text"]) for s in (train, val, test)]
        assert sets[0].isdisjoint(sets[1])
        assert sets[0].isdisjoint(sets[2])
        assert sets[1].isdisjoint(sets[2])

    def test_bad_fractions_raise(self):
        df = self._frame()
        with pytest.raises(ValueError):
            split_data(df, val_frac=0.6, test_frac=0.6)


class FakeTokenizer:
    """Minimal stand-in for an AutoTokenizer, so tests need no weights.

    Returns fixed-length tensors shaped like a real tokenizer's output.
    """

    def __call__(self, text, truncation, padding, max_length, return_tensors):
        assert return_tensors == "pt"
        ids = torch.zeros((1, max_length), dtype=torch.long)
        mask = torch.ones((1, max_length), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}


class TestHateDataset:
    def test_len(self):
        ds = HateDataset(["a", "b", "c"], [0, 1, 0], FakeTokenizer(), max_len=8)
        assert len(ds) == 3

    def test_getitem_shapes_and_label(self):
        ds = HateDataset(["a", "b"], [0, 1], FakeTokenizer(), max_len=8)
        item = ds[1]
        assert item["input_ids"].shape == (8,)  # batch dim squeezed
        assert item["attention_mask"].shape == (8,)
        assert item["labels"].item() == 1
        assert item["labels"].dtype == torch.long

    def test_from_frame(self):
        df = pd.DataFrame({"text": ["x", "y"], "label": [1, 0]})
        ds = HateDataset.from_frame(df, FakeTokenizer(), max_len=4)
        assert len(ds) == 2
        assert ds[0]["input_ids"].shape == (4,)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            HateDataset(["a", "b"], [0], FakeTokenizer())
