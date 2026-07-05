"""Hugging Face Space entrypoint.

Serves the same Gradio interface as `python -m hatedetect.app`, but loads the
checkpoint from ./model (the fine-tuned MuRIL weights are vendored into the Space
at deploy time by scripts/deploy_space.py). Reusing hatedetect.infer.Predictor
means the demo normalizes and tokenizes text exactly the way training did, so
there is no train/serve skew.

Override the checkpoint location with the MODEL_DIR environment variable (handy
for testing this file locally against models/muril-best).
"""

from __future__ import annotations

import os

from hatedetect.app import build_interface
from hatedetect.infer import Predictor

MODEL_DIR = os.environ.get("MODEL_DIR", "model")

predictor = Predictor(MODEL_DIR)
demo = build_interface(predictor)

if __name__ == "__main__":
    demo.launch()
