"""Entrypoint wrapper for LLaVA LoRA fine-tuning.

This script imports LLaVA's trainer directly from the installed `llava` package,
eliminating the need to clone or hardcode the relative path `llava/train/train_mem.py`.
"""

import sys
import traceback

# Only attempt FlashAttention monkey-patch if flash_attn package is actually installed
try:
    import flash_attn
    from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()
    print("[INFO] FlashAttention monkey patch activated.")
except Exception as e:
    print(f"[INFO] Using native PyTorch SDPA attention (flash_attn not loaded: {e})")

try:
    from llava.train.train import train
except Exception:
    print("\n[ERROR] Failed to import llava.train.train. Full traceback below:\n", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

if __name__ == "__main__":
    train()

