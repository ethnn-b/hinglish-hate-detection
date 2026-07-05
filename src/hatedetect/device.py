"""Device selection.

One place so training, evaluation, and inference all agree on where tensors
live. Order of preference: CUDA (the SageMaker box), then Apple MPS (a local
Mac), then CPU. Set HATEDETECT_DEVICE=cpu|mps|cuda to force one when a backend
misbehaves (a few transformer ops are still flaky on MPS).
"""

from __future__ import annotations

import os

import torch


def pick_device() -> str:
    """Return the device string to put the model and batches on."""
    forced = os.environ.get("HATEDETECT_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
