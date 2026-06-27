"""Cross-lingual zero-shot transfer experiment.

For each model preset: fine-tune on the English hate-speech CSV, then evaluate
zero-shot on the Hindi test split (the same held-out examples the in-language
models were scored on). Prints one comparison table and saves the raw numbers to
JSON.

The story this tests: transfer to code-mixed/Indian text should scale with how
much Indian-language pretraining a model had. English-only BERT should collapse
(it cannot even tokenize Devanagari), mBERT and XLM-R should partially transfer,
and MuRIL should transfer best.

    uv run python scripts/run_transfer.py \
        --en-data data/english_hate.csv --hi-data data/hinglish.csv

Each model is trained on English only and never sees a Hindi label, so the Hindi
test score is a true zero-shot-transfer number.
"""

from __future__ import annotations

import argparse
import json

from hatedetect.config import Config
from hatedetect.evaluate import evaluate_checkpoint
from hatedetect.train import train

# Ordered along the pretraining-language gradient, English-only first.
GRADIENT = ["enbert", "mbert", "xlmr", "muril"]
PRETRAIN_NOTE = {
    "enbert": "English only",
    "mbert": "104 languages",
    "xlmr": "100 languages",
    "muril": "Indian languages + transliteration",
}


def run(en_data: str, hi_data: str, presets: list[str], base: Config) -> list[dict]:
    """Train each preset on English, evaluate zero-shot on the Hindi test split."""
    results = []
    for preset in presets:
        print(f"\n{'#' * 70}\n# {preset}: train on English, evaluate zero-shot on Hindi\n{'#' * 70}")
        cfg = Config(
            preset=preset,
            tag="en",
            max_len=base.max_len,
            batch_size=base.batch_size,
            lr=base.lr,
            epochs=base.epochs,
            seed=base.seed,
        )
        # Train on English. Best checkpoint chosen on the English val split.
        train(cfg, en_data)
        # Zero-shot: score the English-trained checkpoint on the Hindi test split.
        metrics = evaluate_checkpoint(str(cfg.output_dir), hi_data, cfg)
        results.append(
            {
                "preset": preset,
                "pretraining": PRETRAIN_NOTE.get(preset, "?"),
                "transfer_macro_f1": metrics["macro_f1"],
                "transfer_accuracy": metrics["accuracy"],
                "transfer_hate_recall": metrics["per_class"]["hate"]["recall"],
            }
        )
    return results


def print_table(results: list[dict]) -> None:
    print(f"\n{'=' * 70}\nZero-shot transfer (English -> Hindi test split)\n{'=' * 70}")
    print(f"{'model':<8} {'pretraining':<36} {'macro-F1':>9} {'hate-rec':>9}")
    for r in results:
        print(
            f"{r['preset']:<8} {r['pretraining']:<36} "
            f"{r['transfer_macro_f1']:>9.4f} {r['transfer_hate_recall']:>9.4f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cross-lingual transfer experiment.")
    parser.add_argument("--en-data", required=True, help="English training CSV")
    parser.add_argument("--hi-data", required=True, help="Hindi CSV (eval on its test split)")
    parser.add_argument(
        "--presets",
        nargs="+",
        default=GRADIENT,
        help=f"models to run, in order (default: {' '.join(GRADIENT)})",
    )
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--max-len", type=int, default=Config.max_len)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--out", default="transfer_results.json", help="where to save raw results")
    args = parser.parse_args()

    base = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        seed=args.seed,
    )
    results = run(args.en_data, args.hi_data, args.presets, base)
    print_table(results)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved raw results to {args.out}")


if __name__ == "__main__":
    main()
