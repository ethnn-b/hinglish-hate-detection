"""Controlled comparison: does MuRIL's edge over XLM-R depend on the script?

Uses the PRISM dataset, which labels each row english / hindi (Devanagari) /
hinglish (Roman-script code-mixed) under one annotation scheme. For each language
subset we fine-tune both models in-language and score them on that subset's test
split, then report the MuRIL minus XLM-R macro-F1 gap per language.

The dataset and labelling are held constant, so the only thing that changes
across the three rows is the script/language. The hypothesis: the gap is small
on English and Devanagari Hindi (where XLM-R is competent) and largest on
Hinglish, the transliterated Roman-script text MuRIL was pretrained for. If that
holds, it isolates transliteration as the source of MuRIL's advantage.

    uv run python scripts/run_script_comparison.py --data-dir data

Expects data/prism_english.csv, data/prism_hindi.csv, data/prism_hinglish.csv
(produced by get_hinglish_data.py).
"""

from __future__ import annotations

import argparse
import json
import shutil

from hatedetect.config import Config
from hatedetect.evaluate import evaluate_checkpoint
from hatedetect.train import train

LANGS = ["english", "hindi", "hinglish"]
SCRIPT_NOTE = {
    "english": "English (Latin)",
    "hindi": "Hindi (Devanagari)",
    "hinglish": "Hinglish (Roman, code-mixed)",
}
PRESETS = ["xlmr", "muril"]


def run(data_dir: str, langs: list[str], base: Config) -> list[dict]:
    """Fine-tune and evaluate each preset in-language on each subset."""
    rows = []
    for lang in langs:
        data_path = f"{data_dir}/prism_{lang}.csv"
        scores = {}
        for preset in PRESETS:
            print(f"\n{'#' * 70}\n# {lang} / {preset}: train and evaluate in-language\n{'#' * 70}")
            cfg = Config(
                preset=preset,
                tag=lang,  # e.g. models/muril-hinglish
                max_len=base.max_len,
                batch_size=base.batch_size,
                lr=base.lr,
                epochs=base.epochs,
                seed=base.seed,
            )
            train(cfg, data_path)
            metrics = evaluate_checkpoint(str(cfg.output_dir), data_path, cfg)
            scores[preset] = metrics["macro_f1"]
            # We only need the metrics; drop the checkpoint so disk does not fill.
            shutil.rmtree(cfg.output_dir, ignore_errors=True)
        rows.append(
            {
                "lang": lang,
                "script": SCRIPT_NOTE.get(lang, lang),
                "xlmr_macro_f1": scores["xlmr"],
                "muril_macro_f1": scores["muril"],
                "muril_minus_xlmr": scores["muril"] - scores["xlmr"],
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    print(f"\n{'=' * 78}\nIn-language MuRIL vs XLM-R by script (same dataset, same labels)\n{'=' * 78}")
    print(f"{'script':<32} {'XLM-R':>8} {'MuRIL':>8} {'MuRIL-XLMR':>12}")
    for r in rows:
        print(
            f"{r['script']:<32} {r['xlmr_macro_f1']:>8.4f} {r['muril_macro_f1']:>8.4f} "
            f"{r['muril_minus_xlmr']:>+12.4f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled script comparison.")
    parser.add_argument("--data-dir", default="data", help="where the prism_<lang>.csv files live")
    parser.add_argument("--langs", nargs="+", default=LANGS, choices=LANGS)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--max-len", type=int, default=Config.max_len)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--out", default="script_comparison_results.json")
    args = parser.parse_args()

    base = Config(epochs=args.epochs, batch_size=args.batch_size, max_len=args.max_len, seed=args.seed)
    rows = run(args.data_dir, args.langs, base)
    print_table(rows)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"saved raw results to {args.out}")


if __name__ == "__main__":
    main()
