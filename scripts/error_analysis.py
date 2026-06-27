"""Error analysis: read a model's mistakes on the held-out test split.

Aggregate metrics (in evaluate.py) tell you how often the model is wrong. This
tells you *where*. It rebuilds the same stratified test split, runs the model,
and prints the misclassified examples split into:

- false positives: true not-hate, predicted hate.
- false negatives: true hate, predicted not-hate.

Within each group it sorts by the model's confidence, most confident first. The
most confident mistakes are the interesting ones: a model that is very sure and
still wrong usually means a genuinely hard example, sarcasm, rare slang, or a
noisy label. If most of the high-confidence errors look ambiguous or mislabeled,
that is a sign you are near the dataset's label-noise ceiling, not leaving easy
accuracy on the table.

    uv run python scripts/error_analysis.py --model models/muril-best --data data/hinglish.csv
"""

from __future__ import annotations

import argparse

from hatedetect.config import Config
from hatedetect.data import load_csv, split_data
from hatedetect.evaluate import LABEL_NAMES
from hatedetect.infer import Predictor


def analyze(model_path: str, data_path: str, cfg: Config, n: int) -> None:
    """Print the test-split mistakes for a saved checkpoint."""
    df = load_csv(data_path)
    _, _, test = split_data(df, cfg.val_frac, cfg.test_frac, cfg.seed)
    texts = test["text"].tolist()
    true = test["label"].tolist()

    predictor = Predictor(model_path, max_len=cfg.max_len)
    preds = predictor.predict(texts)

    # Collect mistakes. Each row: (confidence, text, true_label, pred_label).
    false_pos = []  # true 0, pred 1
    false_neg = []  # true 1, pred 0
    for t, p in zip(true, preds):
        if p["label"] == t:
            continue
        row = (p["prob"], p["text"], t, p["label"])
        if t == 0 and p["label"] == 1:
            false_pos.append(row)
        elif t == 1 and p["label"] == 0:
            false_neg.append(row)

    total = len(true)
    wrong = len(false_pos) + len(false_neg)
    print(f"\nmodel: {model_path}")
    print(f"test examples: {total}, wrong: {wrong} ({wrong / total:.1%})")
    print(f"  false positives (not-hate called hate): {len(false_pos)}")
    print(f"  false negatives (hate called not-hate): {len(false_neg)}")

    def dump(title: str, rows: list) -> None:
        rows.sort(key=lambda r: r[0], reverse=True)  # most confident first
        print(f"\n=== {title} (top {min(n, len(rows))} by confidence) ===")
        for prob, text, t, pred in rows[:n]:
            print(f"  p={prob:.3f}  true={LABEL_NAMES[t]:<9} pred={LABEL_NAMES[pred]:<9} | {text}")

    dump("False positives", false_pos)
    dump("False negatives", false_neg)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a model's test-split mistakes.")
    parser.add_argument("--model", required=True, help="path to a saved checkpoint dir")
    parser.add_argument("--data", required=True, help="path to the text,label CSV")
    parser.add_argument("--n", type=int, default=20, help="how many examples to show per group")
    parser.add_argument("--max-len", type=int, default=Config.max_len)
    parser.add_argument("--seed", type=int, default=Config.seed)
    args = parser.parse_args()

    cfg = Config(max_len=args.max_len, seed=args.seed)
    analyze(args.model, args.data, cfg, args.n)


if __name__ == "__main__":
    main()
