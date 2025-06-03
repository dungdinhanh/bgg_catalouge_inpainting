import argparse
import json
import os
import torch
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
from PIL import Image
from torchvision import transforms
from compel import Compel, ReturnedEmbeddingsType

def parse_args():
    parser = argparse.ArgumentParser(description="Run SDXL ControlNet inference on images")

    # Model paths
    parser.add_argument("--base-model-path", type=str, 
                        default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                        help="Path to the base SDXL model")
    parser.add_argument("--controlnet-path", type=str, 
                        default="/home/ec2-user/ebs-px/SD-diffusers/diffusers/examples/controlnet/data/train_csr/SDXL-nocrop-0228/MODEL_OUT_512/sdxl_controlnet/checkpoint-14000/controlnet",
                        help="Path to the ControlNet model")
    parser.add_argument("--data-root-path", type=str,
                        default="/home/ec2-user/ebs-px/data/ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref/",
                        help="Root path for the data")
    parser.add_argument("--jsonl-path", type=str,
                        default="/home/ec2-user/efs-vio/lpx/lpx-diffuser-edit-files-transfer/test_neutral.jsonl",
                        help="Path to the JSONL file containing image paths and prompts")
    parser.add_argument("--output-dir", type=str,
                        default="./output_images_SDXL_nocrop_0228_14k_512",
                        help="Directory to save output images")
    parser.add_argument("--input-size", type=int,
                        default=768,
                        help="Size to resize input images to")
    parser.add_argument("--num-samples", type=int,
                        default=100,
                        help="Number of samples to process from the JSONL file")
    parser.add_argument("--inference-steps", type=int,
                        default=50,
                        help="Number of inference steps for the diffusion process")
    parser.add_argument("--size-control-scale", type=float,
                        default=-1,
                        help="original size, lower value might lead to background smooth")
    parser.add_argument("--lora-scale", type=float,
                        default=1.0,
                        help="Number of lora scale")
    parser.add_argument("--lora-scale2", type=float,
                        default=1.0,
                        help="Number of lora scale")
    parser.add_argument("--strength", type=float,
                        default=1.0,
                        help="noise strength")
    parser.add_argument("--mask-threshold", type=float,
                        default=0.7,
                        help="Threshold value for mask processing (0.0-1.0)")
    parser.add_argument("--start-color", type=float,
                        default=0,
                        help="the color of masked area (0.0-255.0)")
    parser.add_argument("--guidance-scale", type=float,
                        default=7.5,
                        help="The guidance score of text prompt")
    parser.add_argument("--num-images-per-input", type=int,
                        default=2,
                        help="Number of images to generate for each input with different seeds")
    parser.add_argument("--seed", type=int,
                        default=-1,
                        help="generator seeds")
    parser.add_argument("--save-originals", type=bool,
                        default=True,
                        help="Whether to save original images")
    parser.add_argument("--save-masks", type=bool,
                        default=False,
                        help="Whether to save mask images")
    parser.add_argument("--prompt", type=str,
                        default=None,
                        help="Input text prompt")
    parser.add_argument("--negprompt", type=str,
                        default='Textures, patterns, artifacts, background objects, dark shadows, noise, reflections, clutter, uneven lighting, distortions, color inconsistencies, overexposed or underexposed areas, worst quality, low quality, airbrushed, cartoon, anime, semi-realistic, watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark.',
                        help="Input negative text prompt")
    
    return parser.parse_args()


def resize_with_padding(image, target_size):
    """
    Resize image keeping the aspect ratio and add padding if necessary.
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    width, height = image.size
    aspect = width / height
    
    # Decide if width or height is limiting dimension
    if width > height:
        new_width = target_size[0]
        new_height = int(new_width / aspect)
    else:
        new_height = target_size[1]
        new_width = int(new_height * aspect)
    
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create new image with padding
    new_image = Image.new("RGBA" if image.mode == "RGBA" else "RGB", target_size, (0, 0, 0, 0))
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))
    
    return new_image


def create_masked_image(image, mask, threshold, args):
    """
    Create a masked image based on the threshold value with grey masked areas.
    """
    mask_array = np.asarray(mask).copy() / 255.0
    image_array = np.asarray(image).copy()
    
    # Apply threshold to create binary mask
    binary_mask = np.ones_like(mask_array)
    binary_mask[mask_array >= threshold] = 0
    binary_mask[mask_array < threshold] = 1
    
    # Grey out the masked areas
    grey_value = args.start_color
    masked_image = (1 - binary_mask) * image_array + binary_mask * grey_value

    # Convert everything to Tensors for the pipeline
    masked_image_norm = masked_image / 127.5 - 1.0
    masked_image_tensor = torch.from_numpy(masked_image_norm).float().unsqueeze(0).permute(0, 3, 1, 2)

    # Binary mask for pipeline
    binary_mask_np = binary_mask[:, :, 0, None]  # single channel
    binary_mask_tensor = torch.from_numpy(binary_mask_np).float().unsqueeze(0).permute(0, 3, 1, 2)

    return Image.fromarray(masked_image.astype(np.uint8)), \
           Image.fromarray((binary_mask * 255).astype(np.uint8)), \
           masked_image_tensor, \
           binary_mask_tensor


def setup_directory_structure(output_dir):
    """
    Set up the directory structure for organizing the output images.
    """
    os.makedirs(output_dir, exist_ok=True)
    original_dir = os.path.join(output_dir, "original")
    generated_dir = os.path.join(output_dir, "generated")
    mask_dir = os.path.join(output_dir, "mask")
    
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    return {
        "original": original_dir,
        "generated": generated_dir,
        "mask": mask_dir
    }


def main():
    args = parse_args()
    dirs = setup_directory_structure(args.output_dir)
    
    # Load JSONL file
    records = []
    with open(args.jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            records.append(json.loads(line))
    
    # Load the SDXL Inpainting Pipeline
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.base_model_path, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.load_lora_weights(args.controlnet_path)
    pipe.fuse_lora(lora_scale=args.lora_scale)
    pipe.enable_model_cpu_offload()

    # Build the Compel object for text prompt embeddings
    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True]
    )

    for idx, record in tqdm(enumerate(records), total=len(records), desc="Processing images"):
        image_path = record['image']
        mask_path = record['mask']
        prompt = record['text']

        original_filename = Path(image_path).stem
        pt_name = image_path.split('/')[0]

        # ---------------------------------------------------------------------
        # 1) Load the *original* image before resizing, so we have its size
        # ---------------------------------------------------------------------
        original_image = load_image(os.path.join(args.data_root_path, image_path)).convert("RGB")
        original_width, original_height = original_image.size

        # Now create the "control_image" after resizing & padding
        control_image = resize_with_padding(original_image, args.input_size)

        # Load and resize the mask
        original_mask = load_image(os.path.join(args.data_root_path, mask_path)).convert("RGB")
        mask_image = resize_with_padding(original_mask, args.input_size)

        # Create masked image and mask tensor
        masked_image_pil, binary_mask_pil, masked_image_tensor, binary_mask_tensor = create_masked_image(
            control_image, mask_image, args.mask_threshold, args
        )
        
        if args.save_originals:
            original_save_path = os.path.join(dirs["original"], pt_name, f"{original_filename}.jpg")
            os.makedirs(os.path.join(dirs["original"], pt_name), exist_ok=True)
            original_image.save(original_save_path)

        if args.save_masks:
            mask_save_path = os.path.join(dirs["mask"], pt_name, f"{original_filename}.jpg")
            os.makedirs(os.path.join(dirs["mask"], pt_name), exist_ok=True)
            binary_mask_pil.convert("RGB").save(mask_save_path)

        # Possibly override prompt if you passed `--prompt`
        if args.prompt is not None:
            if 'with neutral tone background' in prompt:
                prompt = prompt.replace('with neutral tone background', args.prompt)
            else:
                prompt = prompt + args.prompt

        # Build prompt embeddings
        conditioning, pooled = compel(prompt)
        negconditioning, negpooled = compel(args.negprompt)
        
        for k in range(args.num_images_per_input):
            if args.seed != -1:
                seed = args.seed
            else:
                seed = idx * 1000 + k
            generator = torch.manual_seed(seed)

            # -----------------------------------------------------------------
            # 2) Call the SDXL pipeline with original_size, crop_size, target_size
            # -----------------------------------------------------------------
            if args.size_control_scale == -1:
                    output_image = pipe(
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=negconditioning,
                    negative_pooled_prompt_embeds=negpooled,
                    image=masked_image_tensor,      # The masked (gray) image
                    mask_image=binary_mask_tensor,  # The mask
                    num_inference_steps=args.inference_steps,
                    generator=generator,
                    strength=args.strength,
                    guidance_scale=args.guidance_scale
                ).images[0]
            else:
                output_image = pipe(
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=negconditioning,
                    negative_pooled_prompt_embeds=negpooled,
                    image=masked_image_tensor,      # The masked (gray) image
                    mask_image=binary_mask_tensor,  # The mask
                    num_inference_steps=args.inference_steps,
                    generator=generator,
                    strength=args.strength,
                    guidance_scale=args.guidance_scale,
                    # Here are the important SDXL kwargs:
                    original_size=(int(original_width*args.size_control_scale), int(original_height*args.size_control_scale)),
                    target_size=(args.input_size, args.input_size),
                    crop_size=(args.input_size, args.input_size),
                    crop_coords_top=0,
                    crop_coords_left=0,
                ).images[0]

            # Resize output_image to match control_image dimensions
            # (Probably 768x768 if that's your args.input_size)
            output_image = output_image.resize(control_image.size, Image.BILINEAR).convert("RGB")

            # Combine masked original with inpainted region
            binary_mask_np = np.array(binary_mask_pil) / 255.0
            inpainted = (1 - binary_mask_np) * np.array(control_image) + binary_mask_np * np.array(output_image)
            inpainted_pil = Image.fromarray(inpainted.astype(np.uint8))

            output_filename = f"{original_filename}_{k}.png"
            output_save_path = os.path.join(dirs["generated"], pt_name, output_filename)
            os.makedirs(os.path.join(dirs["generated"], pt_name), exist_ok=True)
            inpainted_pil.save(output_save_path)

        print(prompt)
        print(f"Processed {idx+1}/{len(records)}: {original_filename} - Generated {args.num_images_per_input} variants")

    print(f"All images have been processed and saved to {args.output_dir}")

if __name__ == "__main__":
    main()
