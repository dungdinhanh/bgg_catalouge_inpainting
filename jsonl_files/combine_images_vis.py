import os
import glob
import numpy as np
import cv2
from PIL import Image
import argparse
from pathlib import Path

def combine_images(image_paths, output_path, max_images_per_row=3):
    """
    Combine images with the same name from different folders into a single image.
    
    Args:
        image_paths: List of paths to images with the same name
        output_path: Path to save the combined image
        max_images_per_row: Maximum number of images per row
    """
    # Read all images
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        images.append(img)
    
    if not images:
        print(f"No valid images found for {os.path.basename(image_paths[0])}")
        return
    
    # Resize all images to the same size (using the first image's dimensions)
    height, width = images[0].shape[:2]
    resized_images = []
    for img in images:
        resized_images.append(cv2.resize(img, (width, height)))
    
    # Calculate rows and columns
    n_images = len(resized_images)
    n_cols = min(n_images, max_images_per_row)
    n_rows = (n_images + max_images_per_row - 1) // max_images_per_row  # Ceiling division
    
    # Create a blank canvas
    canvas = np.ones((height * n_rows, width * n_cols, 3), dtype=np.uint8) * 255
    
    # Place images on canvas
    for i, img in enumerate(resized_images):
        row = i // max_images_per_row
        col = i % max_images_per_row
        y_start = row * height
        y_end = y_start + height
        x_start = col * width
        x_end = x_start + width
        canvas[y_start:y_end, x_start:x_end] = img
    
    # Save the combined image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print(f"Saved combined image to {output_path}")

# def main(input_folders, output_dir, max_images_per_row=3):
#     """
#     Process all images with the same name across different folders.
    
#     Args:
#         input_folders: List of folders containing images
#         output_dir: Directory to save combined images
#         max_images_per_row: Maximum number of images per row
#     """
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Get all image files from the first folder
#     image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
#     first_folder_images = []
#     for ext in image_extensions:
#         first_folder_images.extend(glob.glob(os.path.join(input_folders[0], ext)))
    
#     # Process each image
#     for img_path in first_folder_images:
#         img_name = os.path.basename(img_path)
#         same_name_images = [img_path]
        
#         # Find images with the same name in other folders
#         for folder in input_folders[1:]:
#             matching_img = os.path.join(folder, img_name)
#             if os.path.exists(matching_img):
#                 same_name_images.append(matching_img)
        
#         # Only process if we found the image in at least two folders
#         if len(same_name_images) > 1:
#             output_path = os.path.join(output_dir, f"combined_{img_name}")
#             combine_images(same_name_images, output_path, max_images_per_row)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Combine images with the same name from different folders")
#     parser.add_argument("--input_folders", nargs="+", required=True, help="List of input folders")
#     parser.add_argument("--output_dir", required=True, help="Output directory for combined images")
#     parser.add_argument("--max_per_row", type=int, default=3, help="Maximum images per row")
    
#     args = parser.parse_args()
    
#     main(args.input_folders, args.output_dir, args.max_per_row)


def get_base_name(file_path):
    """
    Get base name without extension and remove '_0' suffix if present.
    """
    base_name = Path(file_path).stem  # Get filename without extension
    if base_name.endswith('_0'):
        base_name = base_name[:-2]  # Remove '_0' suffix
    return base_name

def main(input_folders, output_dir, max_images_per_row=3):
    """
    Process all images with the same base name across different folders,
    regardless of extension and ignoring '_0' suffix.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files from all folders
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    all_images = {}  # Dictionary to store images by base name
    
    # Collect all images from all folders
    for folder in input_folders:
        for ext in image_extensions:
            for img_path in glob.glob(os.path.join(folder, ext)):
                # Get base name without extension and '_0' suffix
                base_name = get_base_name(img_path)
                if base_name not in all_images:
                    all_images[base_name] = []
                all_images[base_name].append(img_path)
    
    # Process each unique base name
    for base_name, image_paths in all_images.items():
        # Only process if we found the image in at least two places
        if len(image_paths) > 3:
            # Use PNG as output format for better quality
            output_path = os.path.join(output_dir, f"combined_{base_name}.png")
            combine_images(image_paths, output_path, max_images_per_row)
            print(f"Processed {len(image_paths)} images for {base_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine images with the same name from different folders")
    parser.add_argument("--input_folders", nargs="+", required=True, help="List of input folders")
    parser.add_argument("--output_dir", required=True, help="Output directory for combined images")
    parser.add_argument("--max_per_row", type=int, default=4, help="Maximum images per row")
    
    args = parser.parse_args()
    main(args.input_folders, args.output_dir, args.max_per_row)

