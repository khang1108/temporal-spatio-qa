#!/bin/bash
# ==============================================================================
# Fine-tune SpaceLLaVA with LoRA on SpatialMQA (ACL 2025)
# Hardware: 1x NVIDIA A100 (40GB or 80GB)
# ==============================================================================

set -eo pipefail

export TRITON_CACHE_DIR="/tmp/triton_${USER:-user}"
mkdir -p "${TRITON_CACHE_DIR}"

# Prevent NCCL P2P / InfiniBand issues on virtualized GPU environments (Thunder Compute, Docker)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Model & Data Paths
MODEL_PATH=${1:-"remyxai/SpaceLLaVA"}
DATA_PATH=${2:-"data/spatial_mqa/train_3780.json"}
IMAGE_FOLDER=${3:-"data/spatial_mqa/images"}
OUTPUT_DIR=${4:-"experiments/checkpoints/spacellava_lora"}
GPU_ID=${5:-"0"}

# DeepSpeed config
DEEPSPEED_CONFIG="scripts/zero2.json"

echo "======================================================================"
echo "Starting LoRA Fine-Tuning for SpaceLLaVA on SpatialMQA"
echo "  Base Model:       ${MODEL_PATH}"
echo "  Training Data:    ${DATA_PATH}"
echo "  Image Folder:     ${IMAGE_FOLDER}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "  GPU ID:           ${GPU_ID}"
echo "  DeepSpeed:        ${DEEPSPEED_CONFIG}"
echo "======================================================================"

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

CUDA_VISIBLE_DEVICES=${GPU_ID} deepspeed --include localhost:${GPU_ID} \
    scripts/finetune/train_mem.py \
    --deepspeed ${DEEPSPEED_CONFIG} \

    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --model_name_or_path ${MODEL_PATH} \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --tf32 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to tensorboard 2>&1 | tee logs/train_spacellava_lora.log

echo "======================================================================"
echo "SpaceLLaVA LoRA fine-tuning completed successfully!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "======================================================================"
