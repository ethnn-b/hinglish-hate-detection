"""Training loop.

Standard full fine-tune: AdamW with a linear warmup-then-decay schedule, plain
cross-entropy (optionally class-weighted), and a small learning rate. After each
epoch we score the validation set and keep the checkpoint with the best
validation macro-F1. Test is never touched here.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import Config
from .data import HateDataset, load_csv, split_data
from .evaluate import compute_metrics, predict, print_report
from .model import build_model, build_tokenizer


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch so runs are reproducible."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: list[int], num_labels: int) -> torch.Tensor:
    """Inverse-frequency weights so minority-class mistakes cost more.

    weight[c] = total / (num_labels * count[c]). A class with half the average
    support gets roughly double the weight.
    """
    counts = np.bincount(labels, minlength=num_labels).astype(float)
    counts[counts == 0] = 1.0  # avoid divide-by-zero on an absent class
    weights = counts.sum() / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float)


def train(cfg: Config, data_path: str) -> dict:
    """Run the full training loop and save the best checkpoint.

    Returns the best validation metrics dict.
    """
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, model: {cfg.model_name}")

    tokenizer = build_tokenizer(cfg.preset)
    model = build_model(cfg.preset, num_labels=cfg.num_labels).to(device)

    df = load_csv(data_path)
    train_df, val_df, _ = split_data(df, cfg.val_frac, cfg.test_frac, cfg.seed)
    print(
        f"rows: {len(df)} total -> {len(train_df)} train, {len(val_df)} val "
        f"(test held out)"
    )
    print(f"train class balance: {np.bincount(train_df['label']).tolist()}")

    train_ds = HateDataset.from_frame(train_df, tokenizer, cfg.max_len)
    val_ds = HateDataset.from_frame(val_df, tokenizer, cfg.max_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    # Loss. Optionally weight by inverse class frequency (computed on train only,
    # never on val/test).
    loss_weight = None
    if cfg.class_weights:
        loss_weight = compute_class_weights(train_df["label"].tolist(), cfg.num_labels).to(device)
        print(f"class weights: {loss_weight.tolist()}")
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    best_macro_f1 = -1.0
    best_metrics: dict = {}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)
        y_true, y_pred = predict(model, val_loader, device)
        metrics = compute_metrics(y_true, y_pred)
        print(
            f"epoch {epoch}/{cfg.epochs}  train_loss={avg_loss:.4f}  "
            f"val_macro_f1={metrics['macro_f1']:.4f}  val_acc={metrics['accuracy']:.4f}"
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_metrics = metrics
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(cfg.output_dir)
            tokenizer.save_pretrained(cfg.output_dir)
            print(f"  saved new best to {cfg.output_dir} (macro-F1 {best_macro_f1:.4f})")

    print_report(best_metrics, title=f"Best validation ({cfg.preset})")
    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a model on the Hinglish CSV.")
    parser.add_argument("--preset", choices=["muril", "xlmr"], default="muril")
    parser.add_argument("--data", required=True, help="path to the text,label CSV")
    parser.add_argument("--max-len", type=int, default=Config.max_len)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="weight the loss by inverse class frequency (for imbalanced data)",
    )
    args = parser.parse_args()

    cfg = Config(
        preset=args.preset,
        max_len=args.max_len,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        class_weights=args.class_weights,
    )
    train(cfg, args.data)


if __name__ == "__main__":
    main()
