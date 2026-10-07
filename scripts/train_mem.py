"""Train LLaVA with native PyTorch SDPA; propagate failures without restarting."""

import inspect
import os


def main():
    os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_{os.environ.get('USER', 'user')}")
    from llava.train.train import train

    if "attn_implementation" not in inspect.signature(train).parameters:
        raise RuntimeError("Installed LLaVA lacks attn_implementation; run scripts/setup_vkong.sh.")
    if os.environ.get("ATTN_IMPL", "sdpa") != "sdpa":
        raise RuntimeError("This entrypoint requires native SDPA (ATTN_IMPL=sdpa).")
    print("[INFO] Requesting native PyTorch SDPA; no external flash-attn monkey patch.", flush=True)
    train(attn_implementation="sdpa")


if __name__ == "__main__":
    main()
