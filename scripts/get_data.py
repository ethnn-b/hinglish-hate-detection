"""Dataset helper: print download links and convert a raw file to text,label CSV.

There is no scraping here. You download a dataset by hand from one of the links
below, then point this script at the raw file to produce the single
`text,label` CSV the pipeline expects.

    uv run python scripts/get_data.py --links
    uv run python scripts/get_data.py --raw data/hasoc_raw.csv --out data/hinglish.csv

Source-label to binary mapping
-------------------------------
HASOC Hindi-English uses a coarse top label `task_1` with values:
  HOF = hate and offensive, NOT = neither.
We map HOF -> 1 and NOT -> 0. That is the whole binary scheme: label 1 is
hate/offensive, label 0 is not. Any finer HASOC sub-labels (HATE, OFFN, PRFN
under task_2) all collapse into 1 because v1 is binary.

For other sources the same idea applies: anything that means hateful, offensive,
abusive, or profane becomes 1; clean/none/normal becomes 0. The mapping is
defined in POSITIVE_LABELS / NEGATIVE_LABELS below and is case-insensitive. If a
file uses values not listed there, the script stops and tells you which ones, so
the mapping stays explicit rather than silently guessing.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

LINKS = {
    "HASOC (Hindi-English code-mixed, primary)": "https://hasocfire.github.io/",
    "Dravidian-CodeMix (second source)": "https://dravidianlangtech.github.io/",
    "GLUECoS (code-switching benchmark, fallback)": "https://microsoft.github.io/GLUECoS/",
}

# Candidate column names, tried in order. First one present in the file wins.
TEXT_COLUMNS = ["text", "tweet", "comment", "content", "sentence"]
LABEL_COLUMNS = ["label", "task_1", "task1", "subtask_a", "category", "class"]

# Case-insensitive value mapping onto the binary scheme.
POSITIVE_LABELS = {"hof", "hate", "offensive", "offn", "prfn", "profane", "abusive", "1", "yes"}
NEGATIVE_LABELS = {"not", "none", "normal", "clean", "non-offensive", "0", "no"}


def print_links() -> None:
    print("Download a dataset by hand from one of these, then convert it:\n")
    for name, url in LINKS.items():
        print(f"  {name}\n    {url}\n")
    print("Then: uv run python scripts/get_data.py --raw <file> --out data/hinglish.csv")


def _pick_column(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    """Return the first candidate column present in df, else raise."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise ValueError(
        f"could not find a {kind} column. Looked for {candidates}, "
        f"file has {list(df.columns)}. Pass --text-col / --label-col to override."
    )


def _map_label(value: object) -> int:
    """Map one raw label value to 0 or 1, or raise if unrecognized."""
    key = str(value).strip().lower()
    if key in POSITIVE_LABELS:
        return 1
    if key in NEGATIVE_LABELS:
        return 0
    raise KeyError(key)


def convert(raw_path: str, out_path: str, text_col: str | None, label_col: str | None) -> None:
    """Read a raw dataset file and write a clean text,label CSV."""
    df = pd.read_csv(raw_path)

    text_col = text_col or _pick_column(df, TEXT_COLUMNS, "text")
    label_col = label_col or _pick_column(df, LABEL_COLUMNS, "label")
    print(f"using text column {text_col!r} and label column {label_col!r}")

    # Map labels, collecting any values that fall outside the known mapping.
    unknown: set[str] = set()

    def safe_map(v: object) -> int | None:
        try:
            return _map_label(v)
        except KeyError as e:
            unknown.add(str(e.args[0]))
            return None

    mapped = df[label_col].map(safe_map)
    if unknown:
        print(
            f"error: unmapped label value(s): {sorted(unknown)}.\n"
            "Add them to POSITIVE_LABELS or NEGATIVE_LABELS in this script, "
            "then rerun.",
            file=sys.stderr,
        )
        sys.exit(1)

    out = pd.DataFrame({"text": df[text_col].astype(str), "label": mapped.astype(int)})
    out = out.dropna().drop_duplicates(subset=["text"])

    out.to_csv(out_path, index=False)
    counts = out["label"].value_counts().sort_index()
    print(f"wrote {len(out)} rows to {out_path}")
    print(f"class balance: not_hate(0)={counts.get(0, 0)}, hate(1)={counts.get(1, 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--links", action="store_true", help="print dataset download links and exit")
    parser.add_argument("--raw", help="path to a raw downloaded dataset file (CSV)")
    parser.add_argument("--out", default="data/hinglish.csv", help="output CSV path")
    parser.add_argument("--text-col", help="override the text column name")
    parser.add_argument("--label-col", help="override the label column name")
    args = parser.parse_args()

    if args.links or not args.raw:
        print_links()
        return

    convert(args.raw, args.out, args.text_col, args.label_col)


if __name__ == "__main__":
    main()
