"""Entrypoint wrapper for LLaVA LoRA fine-tuning.

This script imports LLaVA's trainer directly from the installed `llava` package,
eliminating the need to clone or hardcode the relative path `llava/train/train_mem.py`.
"""

import sys

try:
    from llava.train.train import train
except ImportError:
    print(
        "\n[ERROR] The 'llava' package is not installed in your Python environment.\n"
        "Please install it via:\n"
        "    pip install git+https://github.com/haotian-liu/LLaVA.git\n",
        file=sys.stderr
    )
    sys.exit(1)

if __name__ == "__main__":
    try:
        # Default to FlashAttention-2 if available (standard on A100)
        train(attn_implementation="flash_attention_2")
    except Exception as e:
        print(f"[INFO] FlashAttention-2 init encountered: {e}. Falling back to default attention...")
        train()
