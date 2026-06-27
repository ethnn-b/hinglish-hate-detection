"""Fetch the PRISM code-mixed hate-speech dataset and write per-language CSVs.

PRISM (pankajbiswas6/prism-hinglish-hate-speech on the Hugging Face hub) is a
binary hate-speech dataset with one useful property: it carries a `lang` column
tagging each row as english, hindi (Devanagari), or hinglish (Roman-script
code-mixed). All three share the same binary labels and the same annotation, so
splitting by language gives a controlled comparison where the only thing that
changes is the script/language, not the dataset or labelling scheme.

This writes one text,label CSV per language so the existing train/evaluate
pipeline can run on each:

    uv run python scripts/get_hinglish_data.py            # writes all three
    uv run python scripts/get_hinglish_data.py --langs hinglish

Output files: data/prism_hinglish.csv, data/prism_hindi.csv, data/prism_english.csv

Why it matters here: the earlier Hindi data (HASOC 2019) was native Devanagari,
which does not exercise MuRIL's transliteration pretraining. The hinglish subset
is genuine Roman-script Hinglish ("tum log terrorism ko support karna band
kardo"), which is exactly the setting where MuRIL is expected to beat XLM-R.

Labels are already binary (0 = not hate, 1 = hate), so no remapping is needed.
Provenance is a Kaggle compilation used in a student project, so treat absolute
numbers with some caution; the within-dataset cross-language comparison is the
point, and that holds regardless of absolute label quality.
"""

from __future__ import annotations

import argparse

import pandas as pd

DATASET_ID = "pankajbiswas6/prism-hinglish-hate-speech"
SPLIT_FILES = ["data/train.csv", "data/val.csv", "data/test.csv"]
LANGS = ["hinglish", "hindi", "english"]


def load_full() -> pd.DataFrame:
    """Download all PRISM splits and concatenate into one DataFrame."""
    from huggingface_hub import hf_hub_download

    frames = []
    for fname in SPLIT_FILES:
        path = hf_hub_download(DATASET_ID, fname, repo_type="dataset")
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    if not {"text", "label", "lang"} <= set(df.columns):
        raise ValueError(f"unexpected schema: {list(df.columns)}")
    return df


def write_subsets(langs: list[str], out_dir: str) -> None:
    """Write one text,label CSV per requested language subset."""
    df = load_full()
    print(f"loaded {len(df)} total rows; lang counts: {df['lang'].value_counts().to_dict()}")

    for lang in langs:
        subset = df[df["lang"] == lang][["text", "label"]].dropna()
        subset = subset.drop_duplicates(subset=["text"])
        subset["label"] = subset["label"].astype(int)
        out_path = f"{out_dir}/prism_{lang}.csv"
        subset.to_csv(out_path, index=False)
        counts = subset["label"].value_counts().sort_index()
        print(
            f"wrote {len(subset)} rows to {out_path}  "
            f"(not={counts.get(0, 0)}, hate={counts.get(1, 0)})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--langs", nargs="+", default=LANGS, choices=LANGS, help="which language subsets to write")
    parser.add_argument("--out-dir", default="data", help="directory to write the CSVs into")
    args = parser.parse_args()
    write_subsets(args.langs, args.out_dir)


if __name__ == "__main__":
    main()
