"""Configuration: model presets, hyperparameters, and paths.

One place to look up "what model name does the `muril` preset mean" and "what
learning rate do we train with". Everything else imports from here so the two
training runs (MuRIL and XLM-RoBERTa) stay on identical settings except the
model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Map the short preset names used on the command line to Hugging Face model ids.
# The first two are the headline pair. enbert and mbert are added for the
# cross-lingual transfer study: they span a pretraining-language gradient from
# English-only (bert) through 104-language mBERT to the multilingual XLM-R and
# the India-specific MuRIL.
PRESETS: dict[str, str] = {
    "muril": "google/muril-base-cased",
    "xlmr": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
    "enbert": "bert-base-cased",
}


@dataclass
class Config:
    """Hyperparameters and paths shared across training, eval, and inference.

    Defaults are the sane base-size-encoder settings documented in CLAUDE.md and
    docs/design-decisions.md. Override fields when constructing if you want to
    sweep something by hand.
    """

    # Which model to fine-tune. One of the keys in PRESETS.
    preset: str = "muril"

    # Tokenization and training hyperparameters.
    max_len: int = 128
    batch_size: int = 32
    lr: float = 2e-5
    epochs: int = 4
    num_labels: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42

    # Split fractions. Train gets whatever is left after val and test.
    val_frac: float = 0.1
    test_frac: float = 0.1

    # Whether to weight the loss by inverse class frequency. Off by default
    # (start plain, see design-decisions.md on imbalance handling).
    class_weights: bool = False

    # A label for the checkpoint dir, so runs do not clobber each other. Default
    # "best" keeps the original models/<preset>-best path. The transfer study
    # uses tags like "en" (trained on English) to keep those checkpoints
    # separate from the in-language ones.
    tag: str = "best"

    # Paths. data_dir holds CSVs, models_dir holds saved checkpoints.
    data_dir: Path = field(default_factory=lambda: Path("data"))
    models_dir: Path = field(default_factory=lambda: Path("models"))

    @property
    def model_name(self) -> str:
        """The Hugging Face model id for the chosen preset."""
        if self.preset not in PRESETS:
            valid = ", ".join(sorted(PRESETS))
            raise ValueError(f"unknown preset {self.preset!r}, expected one of: {valid}")
        return PRESETS[self.preset]

    @property
    def output_dir(self) -> Path:
        """Where this run's checkpoint is saved, e.g. models/muril-best or models/muril-en."""
        return self.models_dir / f"{self.preset}-{self.tag}"
