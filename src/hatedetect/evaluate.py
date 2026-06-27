"""Evaluation: metrics and a readable report.

The headline number is macro-F1 (equal weight per class, so the rare class
counts as much as the common one). We also report accuracy, weighted-F1,
per-class precision/recall/F1, and a confusion matrix, because a single averaged
number can hide a class the model is failing on (see docs/concepts.md).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from .config import Config
from .data import HateDataset, load_csv, split_data
from .model import build_model, build_tokenizer

LABEL_NAMES = ["not_hate", "hate"]


def compute_metrics(y_true, y_pred) -> dict:
    """Core metrics from true and predicted label arrays.

    Returns a dict with accuracy, macro-F1, weighted-F1, and the per-class
    report. Used both by `train.py` (to pick the best checkpoint) and here.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


@torch.no_grad()
def predict(model, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader and return (y_true, y_pred) as arrays."""
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(labels.numpy().tolist())
    return np.asarray(all_true), np.asarray(all_pred)


def print_report(metrics: dict, title: str = "Evaluation") -> None:
    """Print metrics in a readable block."""
    print(f"\n=== {title} ===")
    print(f"accuracy    : {metrics['accuracy']:.4f}")
    print(f"macro-F1    : {metrics['macro_f1']:.4f}   <- headline")
    print(f"weighted-F1 : {metrics['weighted_f1']:.4f}")

    print("\nper-class:")
    print(f"  {'class':<10} {'precision':>9} {'recall':>9} {'f1':>9} {'support':>8}")
    for name in LABEL_NAMES:
        row = metrics["per_class"][name]
        print(
            f"  {name:<10} {row['precision']:>9.4f} {row['recall']:>9.4f} "
            f"{row['f1-score']:>9.4f} {int(row['support']):>8}"
        )

    cm = metrics["confusion_matrix"]
    print("\nconfusion matrix (rows = true, cols = predicted):")
    print(f"  {'':<12}{'pred_not':>10}{'pred_hate':>11}")
    print(f"  {'true_not':<12}{cm[0, 0]:>10}{cm[0, 1]:>11}")
    print(f"  {'true_hate':<12}{cm[1, 0]:>10}{cm[1, 1]:>11}")
    print()


def evaluate_checkpoint(model_path: str, data_path: str, cfg: Config) -> dict:
    """Load a saved checkpoint and score it on the held-out test split.

    Rebuilds the exact same split with the same seed as training so the test set
    is the one the model never saw.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = build_tokenizer(model_path)
    model = build_model(model_path, num_labels=cfg.num_labels).to(device)

    df = load_csv(data_path)
    _, _, test = split_data(df, cfg.val_frac, cfg.test_frac, cfg.seed)
    test_ds = HateDataset.from_frame(test, tokenizer, cfg.max_len)
    loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    y_true, y_pred = predict(model, loader, device)
    metrics = compute_metrics(y_true, y_pred)
    print_report(metrics, title=f"Test set: {model_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on the test split.")
    parser.add_argument("--model", required=True, help="path to a saved checkpoint dir")
    parser.add_argument("--data", required=True, help="path to the text,label CSV")
    parser.add_argument("--max-len", type=int, default=Config.max_len)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--seed", type=int, default=Config.seed)
    args = parser.parse_args()

    cfg = Config(
        max_len=args.max_len,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    evaluate_checkpoint(args.model, args.data, cfg)


if __name__ == "__main__":
    main()
