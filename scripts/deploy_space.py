"""Build and push the Gradio demo to a Hugging Face Space.

Assembles a self-contained Space directory in a temp build dir (the app, the
vendored hatedetect package, and the fine-tuned checkpoint) and uploads it. The
Space serves the same interface as `python -m hatedetect.app`.

Auth: set HF_TOKEN to a token with write scope, or run `hf auth login` first.

    # check what would be shipped, no upload, no token needed
    uv run python scripts/deploy_space.py --repo me/hinglish-hate-detector --dry-run

    # create the Space (if needed) and push
    uv run python scripts/deploy_space.py --repo me/hinglish-hate-detector

By default it ships models/muril-best. Pass --model to ship another checkpoint.
The model.safetensors is ~950 MB, uploaded via the Hub's LFS, so the first push
takes a few minutes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACE_SRC = REPO_ROOT / "space"
PKG_SRC = REPO_ROOT / "src" / "hatedetect"


def build(model_dir: Path, build_dir: Path) -> None:
    """Stage the Space contents into build_dir."""
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy2(SPACE_SRC / name, build_dir / name)
    # Vendor the package so `from hatedetect.infer import Predictor` resolves on
    # the Space, which has no editable install of this repo.
    shutil.copytree(
        PKG_SRC, build_dir / "hatedetect", ignore=shutil.ignore_patterns("__pycache__")
    )
    # Ship the checkpoint as ./model, matching the MODEL_DIR default in app.py.
    shutil.copytree(model_dir, build_dir / "model")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo", required=True, help="target Space id, e.g. user/hinglish-hate-detector"
    )
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT / "models" / "muril-demo"),
        help="checkpoint dir to ship (default: models/muril-demo, the all-scripts "
        "PRISM model; pass models/muril-best for the HASOC Devanagari one)",
    )
    parser.add_argument("--private", action="store_true", help="create the Space as private")
    parser.add_argument(
        "--dry-run", action="store_true", help="assemble the build dir but do not upload"
    )
    args = parser.parse_args()

    model_dir = Path(args.model)
    if not (model_dir / "config.json").exists():
        raise SystemExit(
            f"no checkpoint at {model_dir} (expected a config.json). "
            "Train one or pass --model."
        )

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    with tempfile.TemporaryDirectory() as tmp:
        build_dir = Path(tmp) / "space"
        build_dir.mkdir()
        build(model_dir, build_dir)

        print(f"assembled Space in {build_dir}:")
        total = 0
        for p in sorted(build_dir.rglob("*")):
            if p.is_file():
                size = p.stat().st_size
                total += size
                print(f"  {p.relative_to(build_dir)}  ({size / 1e6:.1f} MB)")
        print(f"  total: {total / 1e6:.1f} MB")

        if args.dry_run:
            print("dry run, not uploading")
            return

        if token is None:
            print("note: no HF_TOKEN in env; relying on a prior `hf auth login`")

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        print(f"creating/locating Space {args.repo} ...")
        api.create_repo(
            repo_id=args.repo,
            repo_type="space",
            space_sdk="gradio",
            private=args.private,
            exist_ok=True,
        )
        print("uploading (the ~950 MB checkpoint goes via LFS, this takes a few minutes) ...")
        api.upload_folder(folder_path=str(build_dir), repo_id=args.repo, repo_type="space")
        print(f"done: https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
