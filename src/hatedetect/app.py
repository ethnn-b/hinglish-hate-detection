"""Gradio demo.

A single textbox that runs a saved checkpoint through `infer.Predictor` and
shows the label and confidence. Point it at a checkpoint with --model.
"""

from __future__ import annotations

import argparse

import gradio as gr

from .infer import Predictor

EXAMPLES = [
    "kya bakwaas hai yaar, total waste of time",
    "great work, congratulations on the win!",
    "tum logo ko kuch nahi aata, useless ho sab ke sab",
]


def build_interface(predictor: Predictor) -> "gr.Interface":
    """Build the Gradio interface around a loaded predictor."""

    def classify(text: str) -> dict[str, float]:
        if not text or not text.strip():
            return {}
        result = predictor.predict(text)[0]
        # Show both classes so the confidence split is visible.
        prob_hate = result["prob"] if result["label"] == 1 else 1.0 - result["prob"]
        return {"hate / offensive": prob_hate, "not hate": 1.0 - prob_hate}

    return gr.Interface(
        fn=classify,
        inputs=gr.Textbox(lines=3, label="Hinglish text", placeholder="type something..."),
        outputs=gr.Label(num_top_classes=2, label="prediction"),
        title="Hinglish hate / offensive speech detector",
        description=(
            "Binary classifier for Hindi-English code-mixed text. "
            "Type a sentence and see the model's call."
        ),
        examples=EXAMPLES,
        flagging_mode="never",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Gradio demo.")
    parser.add_argument(
        "--model",
        default="models/muril-best",
        help="path to a saved checkpoint dir (default: models/muril-best)",
    )
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = parser.parse_args()

    predictor = Predictor(args.model, max_len=args.max_len)
    demo = build_interface(predictor)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
