# Background Generation Training Setup

This repository contains scripts and instructions for setting up and training a background generation model using SDXL (Stable Diffusion XL) inpainting.

## Prerequisites

- AWS CLI configured with appropriate permissions
- Python environment
- Git
- Sufficient storage space for training data
- CUDA-compatible GPU

## Complete Setup Commands

```bash
# Create and enter project directory
mkdir lpx
cd lpx

# Download training scripts
mkdir s3_px
aws s3 sync s3://background-generation/diffusers_traing_scripts s3_px
cd s3_px
source setup.sh
cd ..

# Download training datasets
mkdir data
aws s3 cp s3://background-generation/training_data/ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref.v1.1.rm_whitemargin.zip data/ 
aws s3 cp s3://background-generation/training_data/kg09_fashion.zip data/ 
aws s3 cp s3://background-generation/training_data/dpf.zip data/ 

# Extract and organize datasets
cd data
unzip ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref.v1.1.rm_whitemargin.zip
unzip kg09_fashion.zip
unzip dpf.zip

# Merge all image datasets into dpf/images directory
find kg09_fashion/images/ -type f | xargs -I {} mv {} dpf/images/
find ldm_random_crop.myntra40k_amz67k.cleaned_no_whbg.biref.v1.1.rm_whitemargin/images/ -type f | xargs -I {} mv {} dpf/images/
cd ..

# Clone and setup Diffusers
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
cd example/text_to_image
pip install -r requirements_sdxl.txt
pip install bitsandbytes
cd ..
cp -r ~/lpx/s3_px/sdxl-inpainting .
cd sdxl-inpainting
accelerate config
sh train.sh
