#!/bin/bash
# ==============================================================================
# Evaluate Fine-tuned LLaVA-1.5-7B LoRA on SpatialMQA Benchmark (ACL 2025 Long)
# ==============================================================================

set -eo pipefail

GPU_ID=${1:-"0"}
LORA_PATH=${2:-"phuckhangne/llava-1.5-7b-spatialmqa-lora"}
BASE_MODEL=${3:-"liuhaotian/llava-v1.5-7b"}
OUTPUT_JSONL=${4:-"experiments/predictions/llava_1.5_lora_spatialmqa_test.jsonl"}

echo "======================================================================"
echo "Starting Evaluation for LLaVA-1.5-7B LoRA on SpatialMQA Test Set"
echo "  Base Model:       ${BASE_MODEL}"
echo "  LoRA Adapter:     ${LORA_PATH}"
echo "  Test JSONL:       data/spatial_mqa/test.jsonl"
echo "  Image Folder:     data/spatial_mqa/images"
echo "  Output JSONL:     ${OUTPUT_JSONL}"
echo "  GPU ID:           ${GPU_ID}"
echo "======================================================================"

# Ensure output directory exists
mkdir -p "$(dirname "${OUTPUT_JSONL}")"

# Check if images exist, if not download them automatically
if [ ! -d "data/spatial_mqa/images" ] || [ $(ls -1 data/spatial_mqa/images 2>/dev/null | wc -l) -lt 100 ]; then
    echo "[INFO] Images not found or incomplete. Downloading 5,063 images from Hugging Face..."
    python scripts/datasets/download_images.py
fi

# Run inference and compute benchmark metrics with paper comparison
CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/datasets/run_eval.py \
    --model_path "${BASE_MODEL}" \
    --lora_path "${LORA_PATH}" \
    --test_jsonl "data/spatial_mqa/test.jsonl" \
    --image_dir "data/spatial_mqa/images" \
    --output_jsonl "${OUTPUT_JSONL}" \
    --temperature 0.4

echo "======================================================================"
echo "Evaluation completed successfully!"
echo "Raw predictions saved to: ${OUTPUT_JSONL}"
echo "Summary metrics saved to: ${OUTPUT_JSONL%.jsonl}_metrics.json"
echo "======================================================================"
