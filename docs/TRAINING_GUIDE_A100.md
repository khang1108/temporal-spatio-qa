# SpatialMQA: A100 GPU Training & Evaluation Guide
### Re-producing LLaVA-1.5 and SpaceLLaVA LoRA Baselines (ACL 2025)

---

## 1. Hardware & Environment Requirements

- **GPU:** 1x NVIDIA A100 (40GB/80GB), or RTX A6000 (48GB) on Thunder Compute.
- **Runtime:** PyTorch 2.5.1 + CUDA 12.1 wheels; a compatible NVIDIA host driver is required.
- **Python:** 3.10-3.12 in a fresh environment. Thunder images commonly ship
  Python 3.12, so the dependency pins include binary wheels for that version.
- **Attention:** native PyTorch SDPA; no external `flash-attn` or `xformers`.
- Training time must be measured on the actual machine; no A6000 timing has been validated.

### Install Dependencies

Run from the project root on the GPU machine. Keep the provider's preinstalled
Python environment separate, especially if it contains another torch/torchaudio/xformers stack.

```bash
conda create -n spatial-llava python=3.12 -y
conda activate spatial-llava
python -m pip install pip==24.3.1 setuptools==75.6.0 wheel==0.45.1

# Install torch first so DeepSpeed setup can inspect the target runtime.
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
DS_BUILD_OPS=0 python -m pip install -r requirements.txt

# Install only LLaVA source: its upstream metadata otherwise downgrades torch.
python -m pip install --no-deps \
  https://github.com/haotian-liu/LLaVA/archive/c121f0432da27facab705978f83c4ada465e46fd.zip
export TRITON_CACHE_DIR="/tmp/triton_${USER}"
export ATTN_IMPL=sdpa
mkdir -p "$TRITON_CACHE_DIR"
```

`DS_BUILD_OPS=0` skips ahead-of-time DeepSpeed extension builds, not runtime JIT.
If the selected optimizer requires a CUDA extension, a compatible CUDA toolkit
and compiler must also be installed on the GPU machine.

The project intentionally uses torch 2.5.1 instead of upstream LLaVA's declared
2.1.2. Installing LLaVA with `--no-deps` prevents resolver downgrades but does not
rewrite that metadata: `pip check` can report the upstream torch/torchvision
mismatch and missing web-demo dependencies. Do not fix those reports by blindly
installing `llava[train]`; this requirements file targets training/evaluation,
not LLaVA's Gradio server. Runtime compatibility must still be checked.

```bash
python - <<'PYTHON'
import inspect
import torch, torchvision, transformers, deepspeed, peft
from llava.train.train import train
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaSdpaAttention
assert torch.cuda.is_available(), "CUDA GPU unavailable"
assert torch.cuda.is_bf16_supported(), "BF16 unavailable"
assert "attn_implementation" in inspect.signature(train).parameters
config = LlavaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4)
config._attn_implementation = "sdpa"
model = LlavaLlamaForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
assert isinstance(model.model.layers[0].self_attn, LlamaSdpaAttention)
ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
print("SDPA forward/backward passed; run a short actual training check next.")
PYTHON
```

If an earlier install attempted to compile `sentencepiece==0.1.99` or
`scikit-learn==1.2.2`, stop that pip process and rerun the requirements command.
The current pins use CPython 3.12 manylinux wheels and explicitly forbid source
fallback for those two packages; neither CMake nor `libsentencepiece-dev` is
needed.

---

## 2. Dataset Preparation

Ensure the formatted training dataset is generated:
```bash
# Converts data/spatial_mqa/train.jsonl -> data/spatial_mqa/train_3780.json
python src/data/format_llava.py
```

### Download Images (Fast & Recommended - Only 5,063 Images, ~800MB):
Do **NOT** download the full 6.2GB `test2017.zip` from COCO website (heavily throttled to <100 KB/s).
Download only the exact 5,063 images required by SpatialMQA directly from Hugging Face CDN (takes ~1-2 minutes):
```bash
python scripts/download_images.py
# Images will be saved directly into: data/spatial_mqa/images/
```

---

## 3. Launching LoRA Training on A100 (40GB)

### A. Fine-Tune LLaVA-1.5-7B:
```bash
bash scripts/train_llava_lora.sh \
    liuhaotian/llava-v1.5-7b \
    data/spatial_mqa/train_3780.json \
    data/spatial_mqa/images \
    experiments/checkpoints/llava_1.5_7b_lora \
    0
```

### B. Fine-Tune SpaceLLaVA:
```bash
bash scripts/train_spacellava_lora.sh \
    remyxai/SpaceLLaVA \
    data/spatial_mqa/train_3780.json \
    data/spatial_mqa/images \
    experiments/checkpoints/spacellava_lora \
    0
```

### Key Hyperparameters Configured in Script:
- **LoRA Rank ($r$):** 128
- **LoRA Alpha ($\alpha$):** 256
- **LLM Backbone LR:** $2 \times 10^{-4}$
- **Projector LR:** $2 \times 10^{-5}$
- **Batch Size:** 8 per GPU $\times$ 2 grad accum = 16 effective batch size
- **Precision:** BF16 + TF32 enabled
- **DeepSpeed:** ZeRO-2 enabled (`scripts/zero2.json`)
- **Epochs:** 10

---

## 4. Benchmark Evaluation on Test Set (1,076 Samples)

### Evaluate LLaVA-1.5 with Trained LoRA Adapter:
```bash
python scripts/run_eval.py \
    --model_path liuhaotian/llava-v1.5-7b \
    --lora_path experiments/checkpoints/llava_1.5_7b_lora \
    --test_jsonl data/spatial_mqa/test.jsonl \
    --image_dir data/spatial_mqa/images \
    --output_jsonl experiments/predictions/llava_lora_test_preds.jsonl
```

### Evaluate SpaceLLaVA with Trained LoRA Adapter:
```bash
python scripts/run_eval.py \
    --model_path remyxai/SpaceLLaVA \
    --lora_path experiments/checkpoints/spacellava_lora \
    --test_jsonl data/spatial_mqa/test.jsonl \
    --image_dir data/spatial_mqa/images \
    --output_jsonl experiments/predictions/spacellava_lora_test_preds.jsonl
```

---

## 5. Target Benchmark Reproduction Numbers (Paper Table 4 & 5)

| Metric | LLaVA-1.5 (LoRA) | SpaceLLaVA (LoRA) |
|---|---|---|
| **Overall Accuracy** | **46.56%** | **48.14%** |
| Precision (Macro) | 46.85% | 48.60% |
| Recall (Macro) | 46.40% | 47.80% |
| Macro-F1 | 46.10% | 47.70% |
| **Q1 (Out-of-image)** | 53.14% | 54.87% |
| **Q2 (First-person)** | 40.99% | 42.37% |
| **Q3 (Third-person)** | 64.71% | 58.82% |
| **A_x (Horizontal: left/right)** | 55.71% | 56.00% |
| **A_y (Depth: front/behind)** | 29.64% | **51.85%** |
| **A_z (Vertical: above/below)** | 48.13% | 31.41% |
