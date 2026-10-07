"""Push LoRA adapter or full model checkpoint to Hugging Face Hub.

Usage:
    # 1. Login to HF first (if not already logged in):
    huggingface-cli login

    # 2. Upload LoRA Adapter (Recommended, lightweight ~100-300MB):
    python scripts/upload_to_hf.py \
        --folder experiments/checkpoints/llava_1.5_7b_lora \
        --repo_id your-username/llava-1.5-7b-spatialmqa-lora \
        --private
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo


def upload_checkpoint(folder_path: str, repo_id: str, is_private: bool = False):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Checkpoint folder not found: {folder_path}")

    print(f"==================================================")
    print(f"Uploading Checkpoint to Hugging Face Hub")
    print(f"  Source Folder: {folder_path}")
    print(f"  Target Repo:   {repo_id}")
    print(f"  Private Repo:  {is_private}")
    print(f"==================================================")

    api = HfApi()

    # Create repo if not exists
    create_repo(repo_id=repo_id, repo_type="model", private=is_private, exist_ok=True)
    print(f"Repository ready: https://huggingface.co/{repo_id}")

    # Ignore unnecessary huge training logs/cache if any
    ignore_patterns = ["*.pt.tmp*", "zero_to_fp32.py", "checkpoint-*/global_step*"]

    # Upload folder
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=ignore_patterns,
    )

    print(f"\nUpload completed successfully!")
    print(f"View your model at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to Hugging Face Hub")
    parser.add_argument(
        "--folder",
        type=str,
        default="experiments/checkpoints/llava_1.5_7b_lora",
        help="Local checkpoint folder to upload"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Target Hugging Face repo ID, e.g., username/llava-1.5-7b-spatialmqa-lora"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private on Hugging Face"
    )
    args = parser.parse_args()

    upload_checkpoint(
        folder_path=args.folder,
        repo_id=args.repo_id,
        is_private=args.private
    )


if __name__ == "__main__":
    main()
