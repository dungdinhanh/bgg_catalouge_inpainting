import argparse
import json
import os
import torch
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from diffusers import ControlNetModel, StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, UniPCMultistepScheduler, StableDiffusionInpaintPipeline
from diffusers.models import UNet2DConditionModel as UNet
from diffusers import AutoPipelineForInpainting, StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
from PIL import Image
import torch
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
    # parser.add_argument("--data-root-path", type=str,
    #                     default="/home/ec2-user/ebs-px/data/benchmark_2025_pt_extend_v1.raw/SHIRT/",
    #                     help="Root path for the data")
    # parser.add_argument("--jsonl-path", type=str,
    #                     default="/home/ec2-user/efs-vio/lpx/lpx-diffuser-edit-files-transfer/shirt_captions.jsonl",
    #                     help="Path to the JSONL file containing image paths and prompts")
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
    
    Args:
        image: PIL Image to resize
        target_size: tuple or int, target size (width, height) or single value for both dimensions
    
    Returns:
        PIL Image resized with padding
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    # Get original dimensions
    width, height = image.size

    # # Resize so that the smallest dimension matches the target size
    # ratio = max(target_size[0] / width, target_size[1] / height)
    # new_width = int(width * ratio)
    # new_height = int(height * ratio)
    
    # Calculate the aspect ratio
    aspect = width / height
    
    # Calculate new dimensions preserving aspect ratio
    if width > height:
        # Width is the limiting factor
        new_width = target_size[0]
        new_height = int(new_width / aspect)
    else:
        # Height is the limiting factor
        new_height = target_size[1]
        new_width = int(new_height * aspect)
    
    # Resize image according to the calculated dimensions
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create a new image with the target size and paste the resized image
    new_image = Image.new("RGBA" if image.mode == "RGBA" else "RGB", target_size, (0, 0, 0, 0))
    
    # Calculate position to paste (center)
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    
    # Paste the resized image onto the padded image
    new_image.paste(resized_image, (paste_x, paste_y))
    
    return new_image

# def resize_with_padding(image, target_size):
#     """
#     Resize image to match transforms.Resize + transforms.CenterCrop behavior.
#     First resizes so the smallest dimension is at least target_size, then crops to target_size.
    
#     Args:
#         image: PIL Image to resize
#         target_size: tuple or int, target size (width, height) or single value for both dimensions
    
#     Returns:
#         PIL Image resized and cropped to target size
#     """
#     if isinstance(target_size, int):
#         target_size = (target_size, target_size)
    
#     # Get original dimensions
#     width, height = image.size
    
#     # Resize so that the smallest dimension matches or exceeds the target size
#     ratio = max(target_size[0] / width, target_size[1] / height)
#     new_width = int(width * ratio)
#     new_height = int(height * ratio)
    
#     # Resize image using BILINEAR interpolation
#     resized_image = image.resize((new_width, new_height), Image.BILINEAR)
    
#     # Create a new image with the target size
#     new_image = Image.new("RGBA" if image.mode == "RGBA" else "RGB", target_size, (0, 0, 0, 0))
    
#     # Calculate position to paste (center)
#     paste_x = (new_width - target_size[0]) // 2
#     paste_y = (new_height - target_size[1]) // 2
    
#     # Crop the resized image to match target size
#     cropped_image = resized_image.crop((
#         paste_x,
#         paste_y,
#         paste_x + target_size[0],
#         paste_y + target_size[1]
#     ))
    
#     return cropped_image

def create_masked_image(image, mask, threshold, args):
    """
    Create a masked image based on the threshold value with grey masked areas.
    
    Args:
        image: PIL Image of the original image
        mask: PIL Image of the mask
        threshold: float, threshold value for mask processing
        
    Returns:
        PIL Image with the mask applied
    """
    # Convert to numpy arrays
    mask_array = np.asarray(mask).copy() / 255.
    image_array = np.asarray(image).copy()
    
    # Apply threshold to create binary mask
    binary_mask = np.ones_like(mask_array)
    binary_mask[mask_array >= threshold] = 0
    binary_mask[mask_array < threshold] = 1
    
    # Define grey color (128 for medium grey)
    grey_value = args.start_color #128
    
    # Apply mask to image, using grey for masked areas
    masked_image = (1 - binary_mask) * image_array + binary_mask * grey_value
    
    # For the normalized version
    masked_image1 = ((1 - binary_mask) * image_array + binary_mask * grey_value)/127.5-1.0

    masked_image_tensor = torch.from_numpy(masked_image1).float() 
    masked_image_tensor = masked_image_tensor.unsqueeze(0).permute(0, 3, 1, 2)

    binary_mask_tensor = torch.from_numpy(binary_mask)[:,:,0][:,:,None].float() 
    binary_mask_tensor = binary_mask_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    
    # Convert back to PIL Image
    return Image.fromarray(masked_image.astype(np.uint8)), Image.fromarray((binary_mask*255).astype(np.uint8)), masked_image_tensor, binary_mask_tensor


# def create_masked_image(image, mask, threshold):
#     """
#     Create a masked image based on the threshold value.
    
#     Args:
#         image: PIL Image of the original image
#         mask: PIL Image of the mask
#         threshold: float, threshold value for mask processing
        
#     Returns:
#         PIL Image with the mask applied
#     """
#     # Convert to numpy arrays
#     mask_array = np.asarray(mask).copy() / 255.
#     image_array = np.asarray(image).copy()
    
#     # Apply threshold to create binary mask
#     binary_mask = np.ones_like(mask_array)
#     binary_mask[mask_array >= threshold] = 0
#     binary_mask[mask_array < threshold] = 1
    
#     # Apply mask to image
#     masked_image = (1 - binary_mask) * image_array 
#     # masked_image1 = ((1 - binary_mask) * (image_array)+ binary_mask * 255.)/127.5-1.0
#     masked_image1 = ((1 - binary_mask) * (image_array))/127.5-1.0

#     masked_image_tensor = torch.from_numpy(masked_image1).float() 
#     masked_image_tensor = masked_image_tensor.unsqueeze(0).permute(0, 3, 1, 2)

#     binary_mask_tensor = torch.from_numpy(binary_mask)[:,:,0][:,:,None].float() 
#     binary_mask_tensor = binary_mask_tensor.unsqueeze(0).permute(0, 3, 1, 2)
#     # import pdb
#     # pdb.set_trace()
    
#     # Convert back to PIL Image
#     return Image.fromarray(masked_image.astype(np.uint8)), Image.fromarray((binary_mask*255).astype(np.uint8)), masked_image_tensor, binary_mask_tensor

#     # return Image.fromarray(masked_image.astype(np.uint8)), Image.fromarray((binary_mask).astype(np.uint8)), masked_image_tensor, binary_mask_tensor

def setup_directory_structure(output_dir):
    """
    Set up the directory structure for organizing the output images.
    
    Args:
        output_dir: str, base output directory
    
    Returns:
        dict containing paths to subdirectories
    """
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
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
    
    # Set up directory structure
    dirs = setup_directory_structure(args.output_dir)
    
    # Load JSONL file
    records = []
    with open(args.jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:  # Only read the specified number of lines
                break
            records.append(json.loads(line))
    
    # Load models
    # unet = UNet.from_pretrained(args.controlnet_path, torch_dtype=torch.float16)
    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     args.base_model_path, unet=unet, torch_dtype=torch.float16
    # )

    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
    # )

    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
    # )
    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-2-base", torch_dtype=torch.float16
    # )

    # pipe = AutoPipelineForInpainting.from_pretrained(
    #     "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
    #     torch_dtype=torch.float16, variant="fp16")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
        torch_dtype=torch.float16, variant="fp16")
    pipe.load_lora_weights(args.controlnet_path)
    pipe.fuse_lora(lora_scale=args.lora_scale)
    # pipe.load_lora_weights('/home/ec2-user/efs-vio/lpx/lpx-diffuser-edit-files-transfer/ComfyUI/models/checkpoints/lora_chinesearchitecture.safetensors')
    # pipe.fuse_lora(lora_scale=args.lora_scale2)
    # pipe.load_lora_weights('/home/ec2-user/efs-vio/lpx/lpx-diffuser-edit-files-transfer/ComfyUI/models/checkpoints/blur_control_xl_v1.safetensors')
    # pipe.fuse_lora(lora_scale=args.lora_scale2)
    # pipe.load_lora_weights('/home/ec2-user/efs-vio/lpx/lpx-diffuser-edit-files-transfer/ComfyUI/models/checkpoints/sdxl_lightning_2step_lora.safetensors')
    # pipe.fuse_lora(lora_scale=args.lora_scale2)

    # pipe = StableDiffusionXLPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0", 
    #     torch_dtype=torch.float16, variant="fp16")
    # pipe.load_lora_weights(args.controlnet_path)

    # pipe = AutoPipelineForInpainting.from_pretrained(
    #         "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
    #     )
    
    # pipe.load_lora_weights('lora2.safetensors')


    
    # # Speed up diffusion process with faster scheduler and memory optimization
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2] , text_encoder=[pipe.text_encoder, pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

    # compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

    
    # Process each record in the JSONL
    for idx, record in tqdm(enumerate(records), total=len(records), desc="Processing images"):
        # try:
        # Get image path and prompt from the record
        image_path = record['image']
        prompt = record['text']
        mask_path = record['mask']
        
        
        # Extract original filename without extension
        original_filename = Path(image_path).stem
        
        # # Load and resize images
        # control_image = load_image(args.data_root_path + image_path.split('/')[-1]).convert("RGB")
        # control_image = resize_with_padding(control_image, args.input_size)
        
        # mask_image = load_image(args.data_root_path + mask_path.split('/')[-1]).convert("RGB")
        # mask_image = resize_with_padding(mask_image, args.input_size)
        # Load and resize images
        control_image = load_image(args.data_root_path + image_path).convert("RGB")
        control_image = resize_with_padding(control_image, args.input_size)
        
        mask_image = load_image(args.data_root_path + mask_path).convert("RGB")
        mask_image = resize_with_padding(mask_image, args.input_size)
        
        # Create masked image using the threshold
        masked_image, binary_mask, masked_image_tensor, binary_mask_tensor = create_masked_image(control_image, mask_image, args.mask_threshold, args)
        
        # Save original image if requested
        if args.save_originals:
            original_save_path = os.path.join(dirs["original"], f"{original_filename}.jpg")
            control_image.convert("RGB").save(original_save_path)
        
        # Save mask image if requested
        if args.save_masks:
            mask_save_path = os.path.join(dirs["mask"], f"{original_filename}.jpg")
            # masked_image.convert("RGB").save(mask_save_path)
            binary_mask.convert("RGB").save(mask_save_path)
        
        # Generate k images with different seeds
        for k in range(args.num_images_per_input):
            # Use a different seed for each generation
            seed = idx * 1000 + k
            generator = torch.manual_seed(seed)
            
            # # import pdb
            # # pdb.set_trace()

            # # Create transform pipeline
            # transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize([0.5], [0.5])
            # ])

            # # Convert and normalize the image
            # masked_tensor = transform(masked_image)

            # # Add batch dimension if needed
            # if len(masked_tensor.shape) == 3:
            #     masked_tensor = masked_tensor.unsqueeze(0)

            # import pdb
            # pdb.set_trace()
            # Generate image
            # a woman in white pants and a black top with neutral tone background
            if args.prompt is not None:
                if 'with neutral tone background' in prompt:
                # if 'with natural tone background' in prompt:
                    prompt = prompt.replace('with neutral tone background', args.prompt)
                else:
                    prompt = args.prompt

            # # upweight "ball"
            # prompt = "(blurry background, bokeh, depth-of-field)+,The image shows a person wearing a traditional Indian outfit consisting of a white salwar (loose-fitting pleated pants) with a matching dupatta (scarf) and a black sleeveless top. The salwar is adorned with colorful circular and floral motifs in shades of orange, green, and red. It is designed with pleats, creating a voluminous and flowy look, and is cinched at the waist with a visible red zipper. The matching dupatta features the same colorful motifs, with a white base and delicate printed accents. The person is barefoot, and their feet are slightly visible, showing a natural pose. They are also wearing silver anklets. The hand holding the dupatta is adorned with a bracelet, and the nails appear neatly groomed.The background is a neutral gray color, emphasizing the white fabric and colorful patterns. The lighting is soft and even, highlighting the fabric texture and pleats without creating harsh shadows. The overall aesthetic of the outfit is traditional yet modern, with a casual and comfortable appeal."
            conditioning, pooled = compel(prompt)
            negconditioning, negpooled = compel(args.negprompt)
            output_image = pipe(
                prompt_embeds=conditioning, pooled_prompt_embeds=pooled, 
                num_inference_steps=args.inference_steps, 
                generator=generator, 
                image=masked_image_tensor,
                mask_image=binary_mask_tensor,
                # negative_prompt=args.negprompt,
                negative_pooled_prompt_embeds=negpooled,
                negative_prompt_embeds=negconditioning,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
            ).images[0]
            # output_image = pipe(
            #     prompt_embeds=conditioning, pooled_prompt_embeds=pooled, 
            #     num_inference_steps=args.inference_steps, 
            #     generator=generator
            # ).images[0]
            # # prompt="A fashion product image in plain color background with realistic shadows."
            # # output_image = pipe(
            # #     prompt=prompt, 
            # #     num_inference_steps=args.inference_steps, 
            # #     generator=generator, 
            # #     image=masked_image_tensor,
            # #     mask_image=binary_mask_tensor,
            # #     negative_prompt=args.negprompt,
            # #     strength=1
            # # ).images[0]
            

            # conditioning = compel.build_conditioning_tensor(prompt)

            # output_image = pipe(
            #     prompt_embeds=conditioning,
            #     num_inference_steps=args.inference_steps, 
            #     generator=generator, 
            #     image=masked_image_tensor,
            #     mask_image=binary_mask_tensor,
            #     negative_prompt=args.negprompt,
            #     strength=1
            # ).images[0]



            # Resize output_image to match control_image
            output_image = output_image.resize(control_image.size, Image.BILINEAR).convert("RGB")
            
            # Save the generated image
            output_filename = f"{original_filename}_{k}.png"
            output_save_path = os.path.join(dirs["generated"], output_filename)

            binary_mask_np = np.array(binary_mask)/255.
            inpainted = (1 - binary_mask_np)*control_image+binary_mask_np*output_image
            Image.fromarray(inpainted.astype(np.uint8)).save(output_save_path)

        

            # output_image.convert("RGB").save(output_save_path)
        print(prompt)
        print(f"Processed {idx+1}/{len(records)}: {original_filename} - Generated {args.num_images_per_input} variants")
                
        # except Exception as e:
        #     print(f"Error processing record {idx}: {e}")
    
    print(f"All images have been processed and saved to {args.output_dir}")

if __name__ == "__main__":
    main()
