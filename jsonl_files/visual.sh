# python combine_images_vis.py \
#     --input_folders '/home/ec2-user/ebs-px/temp_ldm_model/output_images_cc_seed25_post' '/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/controlnet/output_images_SDXL_nocrop_0228_14k_512_post' '/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/controlnet/output_images_SDXL_nocrop_maskloss_0228_14k_512_post' \
#     --output_dir 'visual_b25m1m2' 

# python combine_images_vis.py \
#     --input_folders './output_images_SDXL_nocrop_maskloss_0228_14k_512' './output_images_SDXL_nocrop_maskloss_0228_14k_512_post' \
#     --output_dir 'visual_SDXL_nocrop_maskloss_0228_post' 

# python combine_images_vis.py \
#     --input_folders '/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/sdxl-inpaint-lora/output_m6-r64-newn_1k_20_512/original' '/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/inpainting_sd2/output_lora_/generated ' '/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/inpainting_sd2/output_lora_3d-render-v2/generated' '/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/inpainting_sd2/output_lora_dongsendongwu/generated' '/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/inpainting_sd2/output_lora_lora_chinesearchitecture/generated' '/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/inpainting_sd2/output_lora_mangheXL/generated' \
#     --output_dir 'visual_sdxl-inpainting-lora' 

# python combine_images_vis.py \
#     --input_folders '/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/sdxl-inpaint-lora/output_m7_bz4_10k_scale0.7_guide5/generated' '/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/sdxl-inpaint-lora/output_m7_bz4_20k_scale0.7_guide5/generated' '/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/sdxl-inpaint-lora/output_m7_bz4_30k_scale0.7_guide5/generated' '/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/sdxl-inpaint-lora/output_m7_bz4_40k_scale0.7_guide5/generated' '/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/sdxl-inpaint-lora/output_m7_bz4_50k_scale0.7_guide5/generated'\
#     --output_dir '/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/sdxl-inpaint-lora/visual_m7_empty_scale0.7_guide5_traingsteps' 

python combine_images_vis.py \
    --input_folders '/home/ec2-user/efs-vio/lpx/lpx-diffuser-edit-files-transfer/output_images_seed_139388700/original' '/home/ec2-user/efs-vio/lpx/lpx-diffuser-edit-files-transfer/output_images_seed_139388700/generated' '/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/sdxl-inpaint-lora/output_m7_bz4_50k/generated' '/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/sdxl-inpaint-lora/output_m7_bz4_50k_scale0.7_guide5/generated' \
    --output_dir '/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/sdxl-inpaint-lora/visual_m7_empty_50k_comp' 