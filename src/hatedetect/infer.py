"""Inference: load a saved checkpoint and predict on raw strings.

Same `normalize()` as training runs first, then the tokenizer and model. Returns
a label (0/1) and the probability of the predicted class. Used by both the CLI
here and the Gradio app.
"""

from __future__ import annotations

import argparse

import torch

from .data import normalize
from .evaluate import LABEL_NAMES
from .model import build_model, build_tokenizer


class Predictor:
    """Holds a loaded model and tokenizer, predicts on strings.

    Construct once and reuse, so the weights are loaded a single time.
    """

    def __init__(self, model_path: str, max_len: int = 128) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = build_tokenizer(model_path)
        self.model = build_model(model_path).to(self.device)
        self.model.eval()
        self.max_len = max_len

    @torch.no_grad()
    def predict(self, texts: str | list[str]) -> list[dict]:
        """Predict on one string or a list. Always returns a list of dicts.

        Each dict has `text` (the original), `label` (0/1), `label_name`, and
        `prob` (probability of the predicted class).
        """
        single = isinstance(texts, str)
        batch_texts = [texts] if single else list(texts)

        cleaned = [normalize(t) for t in batch_texts]
        enc = self.tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)

        results = []
        for i, original in enumerate(batch_texts):
            label = int(pred[i].item())
            results.append(
                {
                    "text": original,
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                    "prob": float(probs[i, label].item()),
                }
            )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict hate/offensive label for input text.")
    parser.add_argument("--model", required=True, help="path to a saved checkpoint dir")
    parser.add_argument("--text", action="append", help="text to classify (repeatable)")
    parser.add_argument("--max-len", type=int, default=128)
    args = parser.parse_args()

    if not args.text:
        parser.error("pass at least one --text")

    predictor = Predictor(args.model, max_len=args.max_len)
    for result in predictor.predict(args.text):
        print(
            f"[{result['label_name']}] (p={result['prob']:.3f})  {result['text']}"
        )


if __name__ == "__main__":
    main()
