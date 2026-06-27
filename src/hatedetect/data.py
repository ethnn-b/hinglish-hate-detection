"""Data pipeline: load a CSV, clean text, split, and tokenize.

The flow is: read a `text,label` CSV, run each row through `normalize()`, split
into train/val/test (stratified, fixed seed), then wrap each split in a
`HateDataset` that tokenizes on demand.

`normalize()` is kept pure (string in, string out, no IO) so it is trivial to
unit test and so the exact same cleaning runs at train and inference time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Precompiled patterns. URLs first, then @mentions, then whitespace runs.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Strip URLs and @mentions, collapse repeated whitespace.

    Light cleaning only. We deliberately keep casing, punctuation, and repeated
    characters because they carry intent ("STUPID", "loooser") and we keep both
    scripts because the code-mixing is the point (see docs/design-decisions.md).

    Pure: no IO, no side effects, same input always gives the same output.
    """
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def load_csv(path: str) -> pd.DataFrame:
    """Load a `text,label` CSV and apply normalization.

    Validates the two required columns exist, drops rows with missing text or
    label, coerces labels to int, and adds a normalized `text` column in place.
    Returns a DataFrame with columns `text` (normalized) and `label` (int).
    """
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} missing required column(s): {sorted(missing)}")

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).map(normalize)
    df["label"] = df["label"].astype(int)

    # Normalization can empty a row (e.g. a tweet that was only a URL). Drop those.
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df


def split_data(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split with a fixed seed.

    Stratifying on the label keeps the class ratio identical across all three
    splits, so the MuRIL vs XLM-R comparison is not biased by a lucky split.
    Returns (train, val, test), each a DataFrame with the same columns as `df`.
    """
    if not 0 < val_frac < 1 or not 0 < test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError(
            f"val_frac ({val_frac}) and test_frac ({test_frac}) must each be in "
            "(0, 1) and sum to less than 1"
        )

    labels = df["label"]
    # First peel off the test set.
    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=labels, random_state=seed
    )
    # Then split the remainder into train and val, keeping val_frac of the
    # ORIGINAL total (not of the remainder), so the fractions mean what they say.
    val_size = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val["label"],
        random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


@dataclass
class HateDataset(Dataset):
    """A torch Dataset that tokenizes text on access.

    The tokenizer is injected (passed in, not constructed here) so tests can
    pass a fake tokenizer instead of downloading model weights, and so the same
    Dataset class works for both MuRIL and XLM-R.
    """

    texts: list[str]
    labels: list[int]
    tokenizer: object  # an AutoTokenizer or any callable matching its API
    max_len: int = 128

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.labels):
            raise ValueError(
                f"texts ({len(self.texts)}) and labels ({len(self.labels)}) "
                "must be the same length"
            )

    @classmethod
    def from_frame(
        cls, df: pd.DataFrame, tokenizer: object, max_len: int = 128
    ) -> "HateDataset":
        """Build a dataset from a DataFrame with `text` and `label` columns."""
        return cls(
            texts=df["text"].tolist(),
            labels=df["label"].tolist(),
            tokenizer=tokenizer,
            max_len=max_len,
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        # The tokenizer returns a batch dim of 1; squeeze it so the DataLoader
        # can collate cleanly into (batch, seq_len).
        item = {key: val.squeeze(0) for key, val in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
