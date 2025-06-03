#!/bin/bash

# Root directory for common paths
ROOT_DIR="/home/ec2-user/ebs-px"

# Model paths
BASE_MODEL_PATH="diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
CONTROLNET_PATH="$ROOT_DIR/SD-diffusers/diffusers/examples/sdxl-inpaint-lora/data/train_csr/SD-M7-empty/MODEL_OUT_768/sdxl_inpainting_lora/checkpoint-50000"

# Data paths
# DATA_ROOT_PATH="$ROOT_DIR/data/ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref"
DATA_ROOT_PATH="$ROOT_DIR/data/benchmark_2025_pt_extend_v1.raw"
JSONL_PATH="/home/ec2-user/efs-vio/lpx/lpx-diffuser-edit-files-transfer/caption_generation_lvm/benchmark_2025_pt_extend_v1_blip.jsonl"

# Output settings
OUTPUT_DIR="./output_m7_50k_scale0.7_guide5_768_size0.25_seedcc_benchmark"
INPUT_SIZE=768
NUM_SAMPLES=10000
INFERENCE_STEPS=50
LORA_SCALE=0.7
LORA_SCALE2=1.0
STRENGTH=1.0
MASK_THRESHOLD=0.7
START_COLOR=0
GUIDANCE_SCALE=5
NUM_IMAGES_PER_INPUT=1
SIZE_CONTROL=0.25
SAVE_ORIGINALS=False
SAVE_MASKS=False
PROMPT=" The image has a high-quality, plain (++), neutral-toned background with a single uniform color, featuring only subtle shadows. Soft, diffused lighting. photorealistic image."
NEGPROMPT="Textures (+), patterns (+), artifacts, background objects, dark shadows, noise, reflections, clutter, uneven lighting, distortions, color inconsistencies, overexposed or underexposed areas, worst quality, low quality, airbrushed, cartoon, anime, semi-realistic, watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark."
SEED=139388700

CUDA_VISIBLE_DEVICES=5 python inference_new.py \
    --base-model-path "$BASE_MODEL_PATH" \
    --controlnet-path "$CONTROLNET_PATH" \
    --data-root-path "$DATA_ROOT_PATH" \
    --jsonl-path "$JSONL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --input-size "$INPUT_SIZE" \
    --num-samples "$NUM_SAMPLES" \
    --inference-steps "$INFERENCE_STEPS" \
    --lora-scale "$LORA_SCALE" \
    --lora-scale2 "$LORA_SCALE2" \
    --strength "$STRENGTH" \
    --mask-threshold "$MASK_THRESHOLD" \
    --start-color "$START_COLOR" \
    --guidance-scale "$GUIDANCE_SCALE" \
    --num-images-per-input "$NUM_IMAGES_PER_INPUT" \
    --save-originals "$SAVE_ORIGINALS" \
    --save-masks "$SAVE_MASKS" \
    --prompt "$PROMPT" \
    --size-control-scale "$SIZE_CONTROL" \
    --negprompt "$NEGPROMPT" \
    --seed "$SEED"
    