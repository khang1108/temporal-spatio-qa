"""
Standalone script to download pre-trained checkpoint weights (.pt) for STKGQA models.

Usage:
    python download_checkpoints.py --model all
    python download_checkpoints.py --model stcqa
    python download_checkpoints.py --model roberta --url <CUSTOM_URL>
"""

import os
import sys
import argparse

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.checkpoint_utils import ensure_checkpoint, DEFAULT_CHECKPOINT_URLS


MODEL_PATHS = {
    "roberta": "experiments/checkpoints/roberta/best_model.pt",
    "multiqa": "experiments/checkpoints/multiqa/best_model.pt",
    "stcqa": "experiments/checkpoints/stcqa/best_model.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download model weights for STKGQA benchmarks.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "roberta", "multiqa", "stcqa"],
        help="Model checkpoint to download (default: all)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Custom download URL (only applies when a single model is specified)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if local checkpoint exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models = ["roberta", "multiqa", "stcqa"] if args.model == "all" else [args.model]

    print(f"Checking checkpoints for: {', '.join(models)}")
    for m in models:
        path = MODEL_PATHS[m]
        url = args.url if (args.model != "all" and args.url) else DEFAULT_CHECKPOINT_URLS.get(m)
        try:
            ensure_checkpoint(
                model_name=m,
                checkpoint_path=path,
                url=url,
                force_download=args.force
            )
            print(f"[{m.upper()}] Checkpoint ready at: {path}")
        except Exception as e:
            print(f"[{m.upper()}] Failed to ensure checkpoint: {e}")


if __name__ == "__main__":
    main()
