# The folder contains FLUX's parameters. It can be created and downloaded automatically when you run the code.
export pretrained_model_name_or_path="diffusers/stable-diffusion-xl-1.0-inpainting-0.1" #"stable-diffusion-v1-5/stable-diffusion-inpainting" #"runwayml/stable-diffusion-inpainting" #"black-forest-labs/FLUX.1-dev"
# No need to change 
export MODEL_TYPE='sdxl_inpainting_lora'
export CONTROL_TYPE='mask' #conditioning_image'
export CAPTION_COLUMN='text'
export TORCH_DISTRIBUTED_DEBUG=INFO
export ROOT_DIR='/home/ec2-user/efs-vio/lpx/lpx-diffuser-edit-files-transfer/'
# The json file path of your training data
export TRAIN_DIR="/home/ec2-user/ebs-px/data/ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref/"
# Define the folder to save downloaded model weights
export CACHE_DIR="./data/train_csr/.cache/huggingface/"
# Define the folder to save your models
export OUTPUT_DIR='./data/train_csr/SD-M7-empty/MODEL_OUT_768/'$MODEL_TYPE

accelerate launch --mixed_precision="fp16" --multi_gpu  train_text_to_image_lora_sdxl.py \
    --pretrained_model_name_or_path=$pretrained_model_name_or_path \
    --train_data_dir=$TRAIN_DIR \
    --jsonl_file=$ROOT_DIR'train_empty.jsonl' \
    --conditioning_image_column=$CONTROL_TYPE \
    --image_column="image" \
    --caption_column=$CAPTION_COLUMN \
    --output_dir=$OUTPUT_DIR \
    --cache_dir=$CACHE_DIR \
    --local_rank=4 \
    --resolution=768 --random_flip \
    --train_batch_size=4 \
    --dataloader_num_workers=2 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --max_train_steps=50000 \
    --checkpointing_steps=5000 \
    --learning_rate=1e-05 \
    --max_grad_norm=1 \
    --mixed_precision="fp16" \
    --validation_epochs=5 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --validation_images $ROOT_DIR"/masked04_51kPy1P0GfL.png" $ROOT_DIR"/masked04_14935598_2.png" \
    --validation_prompts "a woman in a white tank top and black pants with neutral tone background" "a woman wearing a green shirt and jeans with neutral tone background" \
    --validation_masks $ROOT_DIR"/51kPy1P0GfL_mask.png" $ROOT_DIR"/14935598_2_mask.png" 
    # --resume_from_checkpoint $OUTPUT_DIR/checkpoint-35000




