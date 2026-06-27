"""Fetch an English hate-speech dataset and write it as a text,label CSV.

For the cross-lingual transfer study we need an English training set in the same
binary scheme as the Hindi data (label 1 = hate/offensive, 0 = not). This pulls
the Davidson et al. (2017) hate-speech-and-offensive-language dataset from the
Hugging Face hub and maps its three classes onto our binary scheme.

    uv run python scripts/get_english_data.py --out data/english_hate.csv

Source-label mapping
--------------------
Davidson `class`: 0 = hate speech, 1 = offensive language, 2 = neither.
We map 0 and 1 -> 1 (hate/offensive) and 2 -> 0 (not), which matches the binary
definition used for the Hindi data (HOF -> 1, NOT -> 0). Merging hate and
offensive is deliberate: v1 is binary, and it keeps the English and Hindi label
definitions identical so zero-shot transfer is a fair comparison.
"""

from __future__ import annotations

import argparse

import pandas as pd

DATASET_ID = "tdavidson/hate_speech_offensive"


def fetch(out_path: str) -> None:
    """Download Davidson, map to binary, and write a text,label CSV."""
    from datasets import load_dataset

    print(f"loading {DATASET_ID} from the Hugging Face hub...")
    ds = load_dataset(DATASET_ID, split="train")
    df = ds.to_pandas()

    # Be tolerant of column naming across dataset versions.
    text_col = next((c for c in ("tweet", "text") if c in df.columns), None)
    label_col = next((c for c in ("class", "label") if c in df.columns), None)
    if text_col is None or label_col is None:
        raise ValueError(
            f"unexpected schema, columns are {list(df.columns)}; "
            "expected a tweet/text column and a class/label column"
        )

    # 0 hate, 1 offensive -> 1 ; 2 neither -> 0.
    binary = df[label_col].map(lambda c: 0 if int(c) == 2 else 1)
    out = pd.DataFrame({"text": df[text_col].astype(str), "label": binary.astype(int)})
    out = out.dropna().drop_duplicates(subset=["text"])

    out.to_csv(out_path, index=False)
    counts = out["label"].value_counts().sort_index()
    print(f"wrote {len(out)} rows to {out_path}")
    print(f"class balance: not(0)={counts.get(0, 0)}, hate/offensive(1)={counts.get(1, 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/english_hate.csv", help="output CSV path")
    args = parser.parse_args()
    fetch(args.out)


if __name__ == "__main__":
    main()
