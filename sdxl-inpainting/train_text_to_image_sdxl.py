#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion XL for text2image."""

import argparse
import functools
import gc
import logging
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from PIL import Image
from datetime import timedelta

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from datasets import load_from_disk

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
# import boto3
from diffusers.utils import load_image


# # # Initialize S3 client
# session = boto3.Session(profile_name='greenland')
# s3_res = session.resource('s3')
# s3_client = boto3.client('s3')

# # Define your S3 bucket and prefix
# bucket_name = 'background-generation'
# s3_prefix = 'diffusers_traing_scripts/sdxl-inpainting'  # Optional: organize files in S3 with a prefix

# Upload the entire checkpoint directory
def upload_directory_to_s3(local_dir, bucket, s3_path, s3_client):
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Calculate relative path for S3 key
            relative_path = os.path.relpath(local_file_path, local_dir)
            s3_key = os.path.join(s3_path, relative_path)
            
            try:
                s3_client.upload_file(local_file_path, bucket, s3_key)
                logger.info(f"Uploaded {local_file_path} to s3://{bucket}/{s3_key}")
            except Exception as e:
                logger.error(f"Failed to upload {local_file_path}: {str(e)}")


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__)
if is_torch_npu_available():
    import torch_npu

    torch.npu.config.allow_internal_format = False

DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}


def save_model_card(
    repo_id: str,
    images: list = None,
    validation_prompt: str = None,
    base_model: str = None,
    dataset_name: str = None,
    repo_folder: str = None,
    vae_path: str = None,
):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{base_model}** on the **{dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompt: {validation_prompt}: \n
{img_str}

Special VAE used for training: {vae_path}.
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers-training",
        "diffusers",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

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


def create_masked_image(image, mask, threshold):
    """
    Create a masked image based on the threshold value.
    
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
    
    # Apply mask to image
    masked_image = (1 - binary_mask) * image_array
    masked_image1 = (1 - binary_mask) * (image_array/127.5-1.0)

    masked_image_tensor = torch.from_numpy(masked_image1).float() 
    masked_image_tensor = masked_image_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    
    # Convert back to PIL Image
    return Image.fromarray(masked_image.astype(np.uint8)), Image.fromarray((binary_mask*255).astype(np.uint8)), masked_image_tensor




def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--mask_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the masking data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--jsonl_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_masks",
        type=str,
        default=None,
        nargs="+"
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sdxl-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--timestep_bias_strategy",
        type=str,
        default="none",
        choices=["earlier", "later", "range", "none"],
        help=(
            "The timestep bias strategy, which may help direct the model toward learning low or high frequency details."
            " Choices: ['earlier', 'later', 'range', 'none']."
            " The default is 'none', which means no bias is applied, and training proceeds normally."
            " The value of 'later' will increase the frequency of the model's final training timesteps."
        ),
    )
    parser.add_argument(
        "--timestep_bias_multiplier",
        type=float,
        default=1.0,
        help=(
            "The multiplier for the bias. Defaults to 1.0, which means no bias is applied."
            " A value of 2.0 will double the weight of the bias, and a value of 0.5 will halve it."
        ),
    )
    parser.add_argument(
        "--timestep_bias_begin",
        type=int,
        default=0,
        help=(
            "When using `--timestep_bias_strategy=range`, the beginning (inclusive) timestep to bias."
            " Defaults to zero, which equates to having no specific bias."
        ),
    )
    parser.add_argument(
        "--timestep_bias_end",
        type=int,
        default=1000,
        help=(
            "When using `--timestep_bias_strategy=range`, the final timestep (inclusive) to bias."
            " Defaults to 1000, which is the number of timesteps that Stable Diffusion is trained on."
        ),
    )
    parser.add_argument(
        "--timestep_bias_portion",
        type=float,
        default=0.25,
        help=(
            "The portion of timesteps to bias. Defaults to 0.25, which 25% of timesteps will be biased."
            " A value of 0.5 will bias one half of the timesteps. The value provided for `--timestep_bias_strategy` determines"
            " whether the biased portions are in the earlier or later timesteps."
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_npu_flash_attention", action="store_true", help="Whether or not to use npu flash attention."
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")
    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    return args

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds
# # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
# def encode_prompt(batch, text_encoders, tokenizers, proportion_empty_prompts, caption_column, is_train=True):
#     prompt_embeds_list = []
#     prompt_batch = batch[caption_column]

#     captions = []
#     for caption in prompt_batch:
#         if random.random() < proportion_empty_prompts:
#             captions.append("")
#         elif isinstance(caption, str):
#             captions.append(caption)
#         elif isinstance(caption, (list, np.ndarray)):
#             # take a random caption if there are multiple
#             captions.append(random.choice(caption) if is_train else caption[0])

#     with torch.no_grad():
#         for tokenizer, text_encoder in zip(tokenizers, text_encoders):
#             text_inputs = tokenizer(
#                 captions,
#                 padding="max_length",
#                 max_length=tokenizer.model_max_length,
#                 truncation=True,
#                 return_tensors="pt",
#             )
#             text_input_ids = text_inputs.input_ids
#             prompt_embeds = text_encoder(
#                 text_input_ids.to(text_encoder.device),
#                 output_hidden_states=True,
#                 return_dict=False,
#             )

#             # We are only ALWAYS interested in the pooled output of the final text encoder
#             pooled_prompt_embeds = prompt_embeds[0]
#             prompt_embeds = prompt_embeds[-1][-2]
#             bs_embed, seq_len, _ = prompt_embeds.shape
#             prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
#             prompt_embeds_list.append(prompt_embeds)

#     prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
#     pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
#     return {"prompt_embeds": prompt_embeds.cpu(), "pooled_prompt_embeds": pooled_prompt_embeds.cpu()}


def compute_vae_encodings(batch, vae):
    images = batch.pop("pixel_values")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor

    masked_images = batch.pop("conditioning_pixel_values")
    masked_images_values = torch.stack(list(masked_images))
    masked_images_values = masked_images_values.to(memory_format=torch.contiguous_format).float()
    masked_images_values = masked_images_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        masked_images_input = vae.encode(masked_images_values).latent_dist.sample()
    masked_images_input = masked_images_input * vae.config.scaling_factor

    masks = batch.pop("mask_values0")
    mask_values = torch.stack(list(masks))
    mask_values = mask_values.to(memory_format=torch.contiguous_format).float()
    mask_values = mask_values.to(vae.device, dtype=vae.dtype)
    # with torch.no_grad():
    #     model_input = vae.encode(pixel_values).latent_dist.sample()
    # model_input = model_input * vae.config.scaling_factor

    # There might have slightly performance improvement
    # by changing model_input.cpu() to accelerator.gather(model_input)
    return {"model_input": model_input.cpu(), "masked_images_input": masked_images_input.cpu(), "mask_values": mask_values.cpu()}
    # return {"model_input": pixel_values.cpu(), "masked_images_input": masked_images_values.cpu(), "mask_values": mask_values.cpu()}


def generate_timestep_weights(args, num_timesteps):
    weights = torch.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
    else:  # 'none' or any other string
        return weights
    if args.timestep_bias_multiplier <= 0:
        return ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config
    )

    # device = accelerator.device

    # # Optional: for compatibility with NCCL
    # if device.type == "cuda":
    #     torch.cuda.set_device(device)

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # Check for terminal SNR in combination with SNR Gamma
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # Freeze vae and text encoders.
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    # Set unet as trainable.
    unet.train()

    # For mixed precision training we cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)
    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            unet.enable_npu_flash_attention()
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu devices.")
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = unet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, data_dir=args.train_data_dir
        )
    else:
        # data_files = {}
        # if args.train_data_dir is not None:
        #     data_files["train"] = os.path.join(args.train_data_dir, "**")
        # dataset = load_dataset(
        #     "imagefolder",
        #     data_files=data_files,
        #     cache_dir=args.cache_dir,
        # )
        dataset = load_dataset("json", data_files=args.jsonl_file, cache_dir=args.cache_dir)
        dataset = dataset.flatten_indices()
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )
    
    def tokenize_prompt(tokenizer, prompt):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return text_input_ids

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        tokens_one = tokenize_prompt(tokenizer_one, captions)
        tokens_two = tokenize_prompt(tokenizer_two, captions)
        return tokens_one, tokens_two

    class SyncRandomHorizontalFlip(transforms.RandomHorizontalFlip):
        def __init__(self, p=0.5):
            super().__init__(p=p)
            self.flip_state = None
            
        def __call__(self, img):
            """
            Args:
                img (PIL Image): Image to be flipped.
            Returns:
                PIL Image: Randomly flipped image.
            """
            if self.flip_state is None:
                self.flip_state = torch.rand(1) < self.p
            if self.flip_state:
                return TF.hflip(img)
            return img
        
        def reset_state(self):
            self.flip_state = None

    class SyncRandomCrop(transforms.RandomCrop):
        def __init__(self, size):
            super().__init__(size)
            self.cached_params = None
            
        def __call__(self, img):
            """
            Args:
                img (PIL Image): Image to be cropped.
            Returns:
                PIL Image: Randomly cropped image using cached parameters if available.
            """
            if self.cached_params is None:
                # Only generate random parameters the first time
                self.cached_params = self.get_params(img, self.size)
                
            i, j, h, w = self.cached_params
            return TF.crop(img, i, j, h, w)
        
        def reset_state(self):
            """Reset the cached parameters for the next pair of images"""
            self.cached_params = None

    
    # Create synchronized random flip transform
    sync_random_flip = SyncRandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x)
    sync_random_crop = SyncRandomCrop(args.resolution)

    # Modified transform composition
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            sync_random_crop,
            sync_random_flip,
            transforms.ToTensor()
        ]
    )
    # Separate normalization transform to apply after masking
    normalize_transform = transforms.Normalize([0.5], [0.5])

    def preprocess_train(examples):
        # Load and apply initial transforms to images (without normalization)
        images = [
            (image.convert("RGB") if not isinstance(image, str) else Image.open(os.path.join(args.train_data_dir, image)).convert("RGB"))
            for image in examples[args.image_column]
        ]
        # Process conditioning images as before
        conditioning_images = [
            (image.convert("RGB") if not isinstance(image, str) else Image.open(os.path.join(args.mask_data_dir, image)).convert("RGB"))
            for image in examples[args.conditioning_image_column]
        ]
        # Reset the flip state before processing each pair
        sync_random_flip.reset_state()
        sync_random_crop.reset_state()

        original_sizes = [(image.height, image.width) for image in images]

         # Resize images first (keeping transforms separate for clarity)
        resize_transform = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        resized_images = [resize_transform(image) for image in images]
        resized_conditioning_images = [resize_transform(image) for image in conditioning_images]
        
        # Get crop parameters (will be same for both image sets due to sync)
        crop_top_lefts = []
        processed_images = []
        processed_conditioning_images = []
    
        for img, cond_img in zip(resized_images, resized_conditioning_images):
            # Get or use cached crop parameters
            if sync_random_crop.cached_params is None:
                i, j, h, w = sync_random_crop.get_params(img, sync_random_crop.size)
                sync_random_crop.cached_params = (i, j, h, w)
            else:
                i, j, h, w = sync_random_crop.cached_params
                
            # Store crop coordinates
            crop_top_lefts.append((i, j))

            # Apply flip if needed
            if sync_random_flip.flip_state is None:
                sync_random_flip.flip_state = torch.rand(1) < sync_random_flip.p
            
            if sync_random_flip.flip_state:
                img = TF.hflip(img)
                cond_img = TF.hflip(cond_img)
            
            # Apply crop
            img_cropped = TF.crop(img, i, j, h, w)
            cond_img_cropped = TF.crop(cond_img, i, j, h, w)

                
            # Convert to tensor
            img_tensor = transforms.ToTensor()(img_cropped)
            cond_img_tensor = transforms.ToTensor()(cond_img_cropped)
            
            processed_images.append(img_tensor)
            processed_conditioning_images.append(cond_img_tensor)

        
        # # Now both transforms will use the same random state
        # images = [train_transforms(image) for image in images]
        # conditioning_images = [train_transforms(image) for image in conditioning_images]

        # Create masks and apply to unnormalized images
        # Normalize the original images
        images = [normalize_transform(image) for image in processed_images]
        masked_images = []
        masks = []
        for temp_mask, temp_image in zip(processed_conditioning_images, images):
            # mask2 = temp_mask.clone()
   
            # mask = temp_mask.clone()
            # mask[mask2 >= 0.7] = 1
            # mask[mask2 < 0.7] = 0
            mask = (temp_mask >= 0.7).float()
            masked_image = (1 - mask)*temp_image
            # Apply normalization after masking
            masked_images.append(masked_image)
            masks.append(mask)

        
        # masked_images2 = [normalize_transform(image) for image in masked_images]

        # Store all results in examples
        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = masked_images
        examples["mask_values"] = masks
        tokens_one, tokens_two = tokenize_captions(examples)
        examples["input_ids_one"] = tokens_one
        examples["input_ids_two"] = tokens_two
        return examples



    # # Preprocessing the datasets.
    # train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    # train_crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
    # train_flip = transforms.RandomHorizontalFlip(p=1.0)
    # train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    # def preprocess_train(examples):
    #     images = [image.convert("RGB") for image in examples[image_column]]
    #     # image aug
    #     original_sizes = []
    #     all_images = []
    #     crop_top_lefts = []
    #     for image in images:
    #         original_sizes.append((image.height, image.width))
    #         image = train_resize(image)
    #         if args.random_flip and random.random() < 0.5:
    #             # flip
    #             image = train_flip(image)
    #         if args.center_crop:
    #             y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
    #             x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
    #             image = train_crop(image)
    #         else:
    #             y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
    #             image = crop(image, y1, x1, h, w)
    #         crop_top_left = (y1, x1)
    #         crop_top_lefts.append(crop_top_left)
    #         image = train_transforms(image)
    #         all_images.append(image)

    #     examples["original_sizes"] = original_sizes
    #     examples["crop_top_lefts"] = crop_top_lefts
    #     examples["pixel_values"] = all_images
    #     return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    # # Let's first compute all the embeddings so that we can free up the text encoders
    # # from memory. We will pre-compute the VAE encodings too.
    # text_encoders = [text_encoder_one, text_encoder_two]
    # tokenizers = [tokenizer_one, tokenizer_two]
    # compute_embeddings_fn = functools.partial(
    #     encode_prompt,
    #     text_encoders=text_encoders,
    #     tokenizers=tokenizers,
    #     proportion_empty_prompts=args.proportion_empty_prompts,
    #     caption_column=args.caption_column,
    # )
    # compute_vae_encodings_fn = functools.partial(compute_vae_encodings, vae=vae)
    # # save_path = os.path.join(args.output_dir, "precomputed_dataset")

    # # if os.path.exists(save_path):
    # #     print(f"Loading precomputed dataset from {save_path}")
    # #     precomputed_dataset = load_from_disk(save_path)
    # #     precomputed_dataset = precomputed_dataset.with_transform(preprocess_train)
    # # else:
    # with accelerator.main_process_first():
    #     from datasets.fingerprint import Hasher
    #     print(f"[Rank {accelerator.local_process_index}] Mapping dataset with embeddings...")

    #     # fingerprint used by the cache for the other processes to load the result
    #     # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
    #     new_fingerprint = Hasher.hash(args)
    #     new_fingerprint_for_vae = Hasher.hash((vae_path, args))
    #     train_dataset_with_embeddings = train_dataset.map(
    #         compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint
    #     )
    #     train_dataset_with_vae = train_dataset.map(
    #         compute_vae_encodings_fn,
    #         batched=True,
    #         batch_size=args.train_batch_size,
    #         new_fingerprint=new_fingerprint_for_vae,
    #     )
    #     precomputed_dataset = concatenate_datasets(
    #         [train_dataset_with_embeddings, train_dataset_with_vae.remove_columns(["image", "text", "conditioning_image", "mask"])], axis=1
    #     )
    #     # precomputed_dataset.save_to_disk(save_path)
    #     precomputed_dataset = precomputed_dataset.with_transform(preprocess_train)
    #     # save_path = os.path.join(args.output_dir, "precomputed_dataset")
            
    # accelerator.wait_for_everyone()

    # del compute_vae_encodings_fn, compute_embeddings_fn, text_encoder_one, text_encoder_two
    # del text_encoders, tokenizers, vae
    # gc.collect()
    # if is_torch_npu_available():
    #     torch_npu.npu.empty_cache()
    # elif torch.cuda.is_available():
    #     torch.cuda.empty_cache()

    # def collate_fn(examples):
    #     model_input = torch.stack([torch.tensor(example["model_input"]) for example in examples])
    #     original_sizes = [example["original_sizes"] for example in examples]
    #     crop_top_lefts = [example["crop_top_lefts"] for example in examples]
    #     prompt_embeds = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])
    #     pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples])

    #     conditioning_pixel_values = torch.stack([torch.tensor(example["masked_images_input"]) for example in examples])
    #     conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    #     mask_values = torch.stack([torch.tensor(example["mask_values"]) for example in examples])
    #     mask_values = mask_values.to(memory_format=torch.contiguous_format).float()


    #     return {
    #         "model_input": model_input,
    #         "prompt_embeds": prompt_embeds,
    #         "pooled_prompt_embeds": pooled_prompt_embeds,
    #         "original_sizes": original_sizes,
    #         "crop_top_lefts": crop_top_lefts,
    #         "masked_images_input": conditioning_pixel_values,
    #         'mask_values': mask_values
    #     }
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        original_sizes = [example["original_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]
        input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
        input_ids_two = torch.stack([example["input_ids_two"] for example in examples])
        #new added
        conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
        conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

        mask_values = torch.stack([example["mask_values"] for example in examples])
        mask_values = mask_values.to(memory_format=torch.contiguous_format).float()

        result = {
            "pixel_values": pixel_values,
            "input_ids_one": input_ids_one,
            "input_ids_two": input_ids_two,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
            "conditioning_pixel_values": conditioning_pixel_values,
            'mask_values': mask_values
        }
        return result

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        tracker_config.pop("validation_images")
        tracker_config.pop("validation_masks")
        accelerator.init_trackers("text2image-fine-tune-sdxl", config=tracker_config)

    # Function for unwrapping if torch.compile() was used in accelerate.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if torch.backends.mps.is_available() or "playground" in args.pretrained_model_name_or_path:
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    t1 = time.time()
    for epoch in range(first_epoch, args.num_train_epochs):
        # sts_client = boto3.client('sts')
        # ticks = time.perf_counter()
        # assumed_role_object=sts_client.assume_role(
        #     RoleArn="arn:aws:iam::320425221866:role/GreenlandDefaultJobRole",
        #     RoleSessionName=f"DataloaderAssumeRoleSession_{str(ticks)}"
        # )
        # credentials=assumed_role_object['Credentials']
        # s3_client=boto3.client(
        #     's3',
        #     aws_access_key_id=credentials['AccessKeyId'],
        #     aws_secret_access_key=credentials['SecretAccessKey'],
        #     aws_session_token=credentials['SessionToken'],
        # )
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            t2 = time.time()
            with accelerator.accumulate(unet):
                # # Sample noise that we'll add to the latents
                # model_input = batch["model_input"].to(accelerator.device)
                # masked_images_input = batch["masked_images_input"].to(accelerator.device)

                # # images = batch.pop("model_input")
                # pixel_values = batch["model_input"]
                # pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
                # pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
                # with torch.no_grad():
                #     model_input = vae.encode(pixel_values).latent_dist.sample()
                # model_input = model_input * vae.config.scaling_factor

                # masked_images_values = batch["masked_images_input"]
                # masked_images_values = masked_images_values.to(memory_format=torch.contiguous_format).float()
                # masked_images_values = masked_images_values.to(vae.device, dtype=vae.dtype)
                # with torch.no_grad():
                #     masked_images_input = vae.encode(masked_images_values).latent_dist.sample()
                # masked_images_input = masked_images_input * vae.config.scaling_factor

                # # masks = batch.pop("mask_values0")
                # # mask_values = torch.stack(list(masks))
                # # # mask_values = mask_values.to(memory_format=torch.contiguous_format).float()
                # # # mask_values = mask_values.to(vae.device, dtype=vae.dtype)
                # # # # with torch.no_grad():
                # # # #     model_input = vae.encode(pixel_values).latent_dist.sample()
                # # # # model_input = model_input * vae.config.scaling_factor
                # Convert images to latent space
                if args.pretrained_vae_model_name_or_path is not None:
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"]
                # print(pixel_values.shape)
                # exit(0)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    model_input = model_input.to(weight_dtype)


                
                noise = torch.randn_like(model_input)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (model_input.shape[0], model_input.shape[1], 1, 1), device=model_input.device
                    )

                bsz = model_input.shape[0]
                if args.timestep_bias_strategy == "none":
                    # Sample a random timestep for each image without bias.
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                else:
                    # Sample a random timestep for each image, potentially biased by the timestep weights.
                    # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
                    weights = generate_timestep_weights(args, noise_scheduler.config.num_train_timesteps).to(
                        model_input.device
                    )
                    timesteps = torch.multinomial(weights, bsz, replacement=True).long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps).to(dtype=weight_dtype)

                # time ids
                def compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (args.resolution, args.resolution)
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids], device=accelerator.device, dtype=weight_dtype)
                    return add_time_ids

                add_time_ids = torch.cat(
                    [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
                )

                # Predict the noise residual
                unet_added_conditions = {"time_ids": add_time_ids}
                # prompt_embeds = batch["prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
                # pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                # unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders=[text_encoder_one, text_encoder_two],
                    tokenizers=None,
                    prompt=None,
                    text_input_ids_list=[batch["input_ids_one"], batch["input_ids_two"]],
                )
                unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})


                # Convert images to latent space
                if args.pretrained_vae_model_name_or_path is not None:
                    masked_latents = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                else:
                    masked_latents = batch["conditioning_pixel_values"]
                # pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                masked_latents = vae.encode(masked_latents).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    masked_latents = masked_latents.to(weight_dtype)

                # resize mask value to same size with latents
                mask_values = batch["mask_values"]
                if args.pretrained_vae_model_name_or_path is None:
                    mask_values = mask_values.to(weight_dtype)
                # Get the target dimensions from model_pred
                target_height, target_width = noisy_model_input.shape[2], noisy_model_input.shape[3]
                resized_mask = F.interpolate(
                    mask_values,
                    size=(target_height, target_width),
                    mode='nearest'  # Use 'nearest' to preserve binary values (0s and 1s)
                    # Alternative modes: 'bilinear', 'bicubic', 'area'
                )
                concatenated_noisy_latents = torch.cat([noisy_model_input, resized_mask[:,0][:, None], masked_latents], dim=1)
                #"diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
                model_pred = unet(
                    concatenated_noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                elif noise_scheduler.config.prediction_type == "sample":
                    # We set the target to latents here, but the model_pred will return the noise sample prediction.
                    target = model_input
                    # We will have to subtract the noise residual from the prediction to get the target sample.
                    model_pred = model_pred - noise
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # print([t2, t1, t2-t1, step])
                # if t2-t1>600:
                #     # # Assume role and refresh s3 client
                #     sts_client = boto3.client('sts')
                #     ticks = time.perf_counter()
                #     assumed_role_object=sts_client.assume_role(
                #         RoleArn="arn:aws:iam::320425221866:role/GreenlandDefaultJobRole",
                #         RoleSessionName=f"DataloaderAssumeRoleSession_{str(ticks)}"
                #     )
                #     credentials=assumed_role_object['Credentials']
                #     s3_client=boto3.client(
                #         's3',
                #         aws_access_key_id=credentials['AccessKeyId'],
                #         aws_secret_access_key=credentials['SecretAccessKey'],
                #         aws_session_token=credentials['SessionToken'],
                #     )
                #     t1 = time.time()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        # # # Assume role and refresh s3 client
                        # sts_client = boto3.client('sts')
                        # ticks = time.perf_counter()
                        # assumed_role_object=sts_client.assume_role(
                        #     RoleArn="arn:aws:iam::320425221866:role/GreenlandDefaultJobRole",
                        #     RoleSessionName=f"DataloaderAssumeRoleSession_{str(ticks)}"
                        # )
                        # credentials=assumed_role_object['Credentials']
                        # s3_client=boto3.client(
                        #     's3',
                        #     aws_access_key_id=credentials['AccessKeyId'],
                        #     aws_secret_access_key=credentials['SecretAccessKey'],
                        #     aws_session_token=credentials['SessionToken'],
                        # )

                        # # Upload the checkpoint
                        # checkpoint_name = f"checkpoint-{global_step}"
                        # s3_checkpoint_path = f"{s3_prefix}/{checkpoint_name}"
                        # upload_directory_to_s3(save_path, bucket_name, s3_checkpoint_path, s3_client)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)


            if global_step >= args.max_train_steps:
                break
        
        # # accelerator.free_memory()        
        # # gc.collect()
        # # torch.cuda.empty_cache()    
        # accelerator.wait_for_everyone()
        # if accelerator.is_main_process:
        #     if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
        #         logger.info(
        #             f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        #             f" {args.validation_prompts}."
        #         )
        #         if args.use_ema:
        #             # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
        #             ema_unet.store(unet.parameters())
        #             ema_unet.copy_to(unet.parameters())

        #         # create pipeline
        #         vae = AutoencoderKL.from_pretrained(
        #             vae_path,
        #             subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        #             revision=args.revision,
        #             variant=args.variant,
        #         )
        #         pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        #             args.pretrained_model_name_or_path,
        #             vae=vae,
        #             unet=accelerator.unwrap_model(unet),
        #             revision=args.revision,
        #             variant=args.variant,
        #             torch_dtype=weight_dtype,
        #         )
        #         if args.prediction_type is not None:
        #             scheduler_args = {"prediction_type": args.prediction_type}
        #             pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)

        #         pipeline = pipeline.to(accelerator.device)
        #         pipeline.set_progress_bar_config(disable=True)

        #         # run inference
        #         generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
        #         # pipeline_args = {"prompt": args.validation_prompt}

        #         # with autocast_ctx:
        #         #     images = [
        #         #         pipeline(**pipeline_args, generator=generator, num_inference_steps=25).images[0]
        #         #         for _ in range(args.num_validation_images)
        #         #     ]
        #         if len(args.validation_images) == len(args.validation_prompts):
        #             validation_images = args.validation_images
        #             validation_prompts = args.validation_prompts
        #             validation_masks = args.validation_masks
        #         elif len(args.validation_images) == 1:
        #             validation_images = args.validation_images * len(args.validation_prompts)
        #             validation_prompts = args.validation_prompts
        #             validation_masks = args.validation_masks
        #         elif len(args.validation_prompts) == 1:
        #             validation_images = args.validation_images
        #             validation_prompts = args.validation_prompts * len(args.validation_images)
        #             validation_masks = args.validation_masks
        #         else:
        #             raise ValueError(
        #                 "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        #             )

        #         images = []
        #         for i in range(len(args.validation_prompts)):
        #             if torch.backends.mps.is_available():
        #                 autocast_ctx = nullcontext()
        #             else:
        #                 autocast_ctx = torch.autocast(accelerator.device.type)

        #             validation_image = load_image(validation_images[i])
        #             validation_mask = load_image(validation_masks[i])
        #             validation_image = resize_with_padding(validation_image, 768)
        #             validation_mask = resize_with_padding(validation_mask, 768)
        #             # Create masked image using the threshold
        #             masked_image, binary_mask, masked_image_tensor = create_masked_image(validation_image, validation_mask, 0.7)
                    

        #             with autocast_ctx:
        #                 image = pipeline(
        #                     args.validation_prompts[i], 
        #                     num_inference_steps=20, 
        #                     image=masked_image,
        #                     mask_image=binary_mask,
        #                     generator=generator,
        #                     strength=1).images[0]
        #                 if not os.path.isdir('val_visual'):
        #                     os.mkdir('val_visual')
        #                 # Save the generated image
        #                 output_filename = f"val_visual/{i}.jpg"
        #                 image.convert("RGB").save(output_filename)
                    
        #             images.append(image)

        #         for tracker in accelerator.trackers:
        #             if tracker.name == "tensorboard":
        #                 np_images = np.stack([np.asarray(img) for img in images])
        #                 tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        #             if tracker.name == "wandb":
        #                 tracker.log(
        #                     {
        #                         "validation": [
        #                             wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
        #                             for i, image in enumerate(images)
        #                         ]
        #                     }
        #                 )

        #         del pipeline
        #         if is_torch_npu_available():
        #             torch_npu.npu.empty_cache()
        #         elif torch.cuda.is_available():
        #             torch.cuda.empty_cache()

        #         if args.use_ema:
        #             # Switch back to the original UNet parameters.
        #             ema_unet.restore(unet.parameters())

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        # Serialize pipeline.
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            vae=vae,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        if args.prediction_type is not None:
            scheduler_args = {"prediction_type": args.prediction_type}
            pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
        pipeline.save_pretrained(args.output_dir)

        # run inference
        images = []
        if args.validation_prompts and args.num_validation_images > 0:
            pipeline = pipeline.to(accelerator.device)
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

            with autocast_ctx:
                images = [
                    pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                    for _ in range(args.num_validation_images)
                ]

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                for i, image in enumerate(images)
                            ]
                        }
                    )

        if args.push_to_hub:
            save_model_card(
                repo_id=repo_id,
                images=images,
                validation_prompt=args.validation_prompt,
                base_model=args.pretrained_model_name_or_path,
                dataset_name=args.dataset_name,
                repo_folder=args.output_dir,
                vae_path=args.pretrained_vae_model_name_or_path,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
