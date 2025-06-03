# The folder contains FLUX's parameters. It can be created and downloaded automatically when you run the code.
export pretrained_model_name_or_path="diffusers/stable-diffusion-xl-1.0-inpainting-0.1" #"stable-diffusion-v1-5/stable-diffusion-inpainting" #"runwayml/stable-diffusion-inpainting" #"black-forest-labs/FLUX.1-dev"
# No need to change 
export MODEL_TYPE='sdxl_inpainting'
export CONTROL_TYPE='mask' #conditioning_image'
export CAPTION_COLUMN='text'
export TORCH_DISTRIBUTED_DEBUG=INFO
# The json file dir of your training data
export ROOT_DIR='/home/greenland-user/lpx/s3_px/jsonl_files/'
# The image folder of your training data
export TRAIN_DIR="/home/greenland-user/lpx/data/data/dpf/"
# Define the folder to save downloaded model weights
export CACHE_DIR="./data/train_csr/.cache/huggingface/"
# Define the folder to save your models
export OUTPUT_DIR='./data/train_csr/SD-M10/MODEL_OUT_768/'$MODEL_TYPE
export TORCH_DISTRIBUTED_TIMEOUT=3600
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # for more info if it happens again



accelerate launch \
    train_text_to_image_sdxl.py \
    --pretrained_model_name_or_path=$pretrained_model_name_or_path \
    --train_data_dir=$TRAIN_DIR \
    --jsonl_file=$ROOT_DIR'/train_ldm_dpf_kg09_empty.jsonl' \
    --conditioning_image_column=$CONTROL_TYPE \
    --image_column="image" \
    --caption_column=$CAPTION_COLUMN \
    --output_dir=$OUTPUT_DIR \
    --cache_dir=$CACHE_DIR \
    --resolution=768 --random_flip \
    --train_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --max_train_steps=80000 \
    --checkpointing_steps=2000 \
    --learning_rate=1e-06 \
    --max_grad_norm=1 \
    --use_8bit_adam \
    --validation_epochs=1 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --validation_images $ROOT_DIR"/masked04_51kPy1P0GfL.png" $ROOT_DIR"/masked04_14935598_2.png" \
    --validation_prompts "a woman in a white tank top and black pants with neutral tone background" "a woman wearing a green shirt and jeans with neutral tone background" \
    --validation_masks $ROOT_DIR"/masked04_51kPy1P0GfL.png" $ROOT_DIR"/masked04_14935598_2.png" 
    # --resume_from_checkpoint $OUTPUT_DIR/checkpoint-44000  
    #--multi_gpu
    # --jsonl_file=$ROOT_DIR'/filtered_train_empty.jsonl' \
    




