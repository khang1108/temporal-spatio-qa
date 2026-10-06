"""Entrypoint wrapper for LLaVA LoRA fine-tuning.

This script imports LLaVA's trainer directly from the installed `llava` package,
eliminating the need to clone or hardcode the relative path `llava/train/train_mem.py`.
"""

import inspect
import os
import sys
import traceback

# Check if flash_attn package is available
has_flash_attn = False
try:
    import flash_attn
    from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()
    print("[INFO] FlashAttention monkey patch activated.")
    has_flash_attn = True
except Exception as e:
    print(f"[INFO] flash_attn not loaded ({e}). Using PyTorch native SDPA kernel.")

try:
    from llava.train.train import train
except Exception:
    print("\n[ERROR] Failed to import llava.train.train. Full traceback below:\n", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

if __name__ == "__main__":
    sig = inspect.signature(train)
    if "attn_implementation" in sig.parameters:
        attn_backend = os.environ.get("ATTN_IMPL", "flash_attention_2" if has_flash_attn else "sdpa")
        print(f"[INFO] Launching train(attn_implementation='{attn_backend}')")
        try:
            train(attn_implementation=attn_backend)
        except Exception as err:
            print(f"[WARNING] train(attn_implementation='{attn_backend}') failed: {err}. Falling back to default train().")
            train()
    else:
        print("[INFO] Calling default train()")
        train()

