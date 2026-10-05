# SpatialMQA: A100 GPU Training & Evaluation Guide
### Re-producing LLaVA-1.5 and SpaceLLaVA LoRA Baselines (ACL 2025)

---

## 1. Hardware & Environment Requirements

- **GPU:** 1x NVIDIA A100 (40GB or 80GB)
- **CUDA:** 12.1+ / 11.8+
- **Python:** 3.10 - 3.12
- **Estimated Training Time:**
  - LLaVA-1.5-7B (10 epochs, 3,780 samples): **~20 - 25 minutes**
  - SpaceLLaVA (10 epochs): **~20 - 25 minutes**

### Install Dependencies:
```bash
# 1. Install PyTorch with CUDA support (if not already installed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install LLaVA, DeepSpeed, and FlashAttention
pip install deepspeed==0.14.2
pip install git+https://github.com/haotian-liu/LLaVA.git
pip install flash-attn --no-build-isolation

# 3. Install core packages
pip install peft==0.9.0 transformers==4.37.2 accelerate==0.21.0
pip install pillow pandas numpy tqdm
```

---

## 2. Dataset Preparation

Ensure the formatted training dataset is generated:
```bash
# Converts data/spatial_mqa/train.jsonl -> data/spatial_mqa/train_3780.json
python src/data/format_llava.py
```

### Download COCO2017 Test Images:
The paper uses images from COCO2017 test set (`http://images.cocodataset.org/test2017/`):
```bash
# Option A: Download full COCO test2017 (recommended for cluster)
mkdir -p data/COCO2017 && cd data/COCO2017
wget http://images.cocodataset.org/zips/test2017.zip
unzip -q test2017.zip && cd ../..
```

---

## 3. Launching LoRA Training on A100 (40GB)

### A. Fine-Tune LLaVA-1.5-7B:
```bash
bash scripts/train_llava_lora.sh \
    liuhaotian/llava-v1.5-7b \
    data/spatial_mqa/train_3780.json \
    data/COCO2017/test2017 \
    experiments/checkpoints/llava_1.5_7b_lora \
    0
```

### B. Fine-Tune SpaceLLaVA:
```bash
bash scripts/train_spacellava_lora.sh \
    remyxai/SpaceLLaVA \
    data/spatial_mqa/train_3780.json \
    data/COCO2017/test2017 \
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
    --image_dir data/COCO2017/test2017 \
    --output_jsonl experiments/predictions/llava_lora_test_preds.jsonl
```

### Evaluate SpaceLLaVA with Trained LoRA Adapter:
```bash
python scripts/run_eval.py \
    --model_path remyxai/SpaceLLaVA \
    --lora_path experiments/checkpoints/spacellava_lora \
    --test_jsonl data/spatial_mqa/test.jsonl \
    --image_dir data/COCO2017/test2017 \
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
