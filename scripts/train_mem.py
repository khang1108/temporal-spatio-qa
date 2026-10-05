"""Entrypoint wrapper for LLaVA LoRA fine-tuning.

This script imports LLaVA's trainer directly from the installed `llava` package,
eliminating the need to clone or hardcode the relative path `llava/train/train_mem.py`.
"""

import sys

# Optional FlashAttention monkey-patch (only if flash_attn is installed)
try:
    from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()
    print("[INFO] FlashAttention monkey patch activated.")
except Exception as e:
    print(f"[INFO] Running with native PyTorch SDPA attention (flash_attn package not loaded: {e})")

try:
    from llava.train.train import train
except ImportError:
    print(
        "\n[ERROR] The 'llava' package is not installed in your Python environment.\n"
        "Please install it via:\n"
        "    pip install git+https://github.com/haotian-liu/LLaVA.git --no-deps\n",
        file=sys.stderr
    )
    sys.exit(1)

if __name__ == "__main__":
    train()

