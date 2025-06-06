# from powerpaint.datasets.mask_generator.masks_seg import *
# from powerpaint.datasets.catalogue import *

# import os
# from PIL import Image

# from torch.utils.data import DataLoader
# from torchvision import transforms


import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset

from masks_seg import *
# from powerpaint.datasets.catalogue import *
from PIL import Image, ImageFile
import numpy as np
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True


def augment_images(image, mask, resolution):
    mask = Image.fromarray((mask.squeeze(0) * 255).astype("uint8")).convert("L")

    resize = transforms.Resize((resolution, resolution))
    image, mask = resize(image), resize(mask)

    crop = transforms.RandomCrop(resolution)
    image, mask = crop(image), crop(mask)

    

    to_tensor = transforms.ToTensor()
    image = to_tensor(image)
    mask = to_tensor(mask)
    mask[mask != 0] = 1

    normalize = transforms.Normalize([0.5], [0.5])
    image = normalize(image)

    return image, mask


class MaskGenerateDataset(Dataset):
    def __init__(self, image_dir, seg_dir, mask_dir, file_list, resolution=512 ,transform=None, device="cpu"):
        """
        Args:
            image_dir (str): Directory with all the images.
            labels_dict (dict): Dictionary mapping image filenames to labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.mask_dir = mask_dir
        os.makedirs(self.mask_dir, exist_ok=True)
        self.file_list = file_list
        self.transform = transform
        f = open(self.file_list, "r")
        rel_img_paths = f.readlines()
        self.img_paths = []
        self.rel_img_paths = []
        self.resolution= resolution
        for rel_img_path in rel_img_paths:
            rel_img_path = os.path.normpath(rel_img_path.strip())
            img_path = os.path.join(image_dir, rel_img_path)
            self.rel_img_paths.append(rel_img_path)
            self.img_paths.append(img_path)
        self.mask_gen = RandomRectangleMaskGenerator()
        self.device = device
        self.img_error_file = os.path.join(mask_dir, "failure_img.txt")
        f = open(self.img_error_file, "w")
        f.write("")
        f.close()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        rel_path = self.rel_img_paths[idx]
        mask_path = os.path.join(self.mask_dir, rel_path)
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        # img_path = os.path.join(self.image_dir, img_name)
        # image = Image.open(img_path).convert('RGB')
        try:
            image = Image.open(img_path).convert("RGB")
        except OSError:
            print(f"Warning: could not open image {img_path}")
            print(f"Initialize image with all ones")
            image =  np.ones((self.resolution, self.resolution, 3))  # or handle as needed
            f = open(self.img_error_file, "a")
            f.write(f"{img_path}\n")
            f.close()

        
        
        # seg = torch.from_numpy(seg).to(self.pipeline.device)
        # seg = torch.from_numpy(seg).type(torch.float16).cuda()
        image_np = np.array(image)
        gen_mask = self.mask_gen(image_np, raw_image=rel_path)
        if self.transform:
            image = self.transform(image)

        # Convert to CPU NumPy array
        gen_mask_np = np.squeeze(gen_mask)

        # Ensure values are 0 or 255
        try:
            gen_mask_img = Image.fromarray((gen_mask_np * 255).astype(np.uint8))
        except:
            print(gen_mask_np.shape)
            print(rel_path)
            exit(0)
        gen_mask_img.save(mask_path)
        return image

# Include your MaskGenerateDataset class definition here (unchanged from your latest)


def run():
    print("Mask generating")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = MaskGenerateDataset(
        "/home/ubuntu/inpaint_full/",
        "/home/ubuntu/inpaint_seg/images_segment",
        "/opt/dlami/nvme/inpainting/mask_random_min30_max100",
        "/home/ubuntu/inpaint_full/train_cleaned.txt",
        transform=transform
    )

    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    # dataloader = DataLoader(dataset, batch_size=1000, shuffle=False, sampler=sampler, num_workers=8)
    dataloader = DataLoader(
                dataset,
                batch_size=1000,
                shuffle=False,      # Enable shuffling since we're not using a sampler
                num_workers=8
            )
    # print(f"Rank {rank} on device {torch.cuda.current_device()} gets {len(sampler)} samples")
    # exit(0)
    count = 0
    for image in dataloader:
        
        if count % 5000 == 0:
            print(f"Complete {count} over {len(dataset)}")
        count += 1000
        pass  # All saving happens in __getitem__, so we just iterate

def main():
    run()

if __name__ == "__main__":
    main()

# def augment_images(image, mask, resolution):
#     mask = Image.fromarray((mask.squeeze(0) * 255).astype("uint8")).convert("L")

#     resize = transforms.Resize((resolution, resolution))
#     image, mask = resize(image), resize(mask)

#     crop = transforms.RandomCrop(resolution)
#     image, mask = crop(image), crop(mask)

    

#     to_tensor = transforms.ToTensor()
#     image = to_tensor(image)
#     mask = to_tensor(mask)
#     mask[mask != 0] = 1

#     normalize = transforms.Normalize([0.5], [0.5])
#     image = normalize(image)

#     return image, mask


# class MaskGenerateDataset(Dataset):
#     def __init__(self, image_dir, seg_dir, mask_dir, file_list, scheme_ratio=0.5, resolution=512 ,transform=None):
#         """
#         Args:
#             image_dir (str): Directory with all the images.
#             labels_dict (dict): Dictionary mapping image filenames to labels.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.image_dir = image_dir
#         self.seg_dir = seg_dir
#         self.mask_dir = mask_dir
#         os.makedirs(self.mask_dir, exist_ok=True)
#         self.file_list = file_list
#         self.transform = transform
#         self.scheme_ratio = scheme_ratio
#         f = open(self.file_list, "r")
#         rel_img_paths = f.readlines()
#         self.seg_paths = []
#         self.img_paths = []
#         self.rel_img_paths = []
#         self.resolution= resolution
#         for rel_img_path in rel_img_paths:
#             rel_img_path = os.path.normpath(rel_img_path.strip())
#             img_path = os.path.join(image_dir, rel_img_path)
#             seg_img_path = os.path.join(seg_dir, rel_img_path.split(".")[0] + ".png")
#             self.rel_img_paths.append(rel_img_path)
#             self.img_paths.append(img_path)
#             self.seg_paths.append(seg_img_path)
#         self.mask_gen_a = RandomRectangleMaskWithSegmGenerator()
#         self.mask_gen_b = RandomRectangleMaskWithSegmOverlapGenerator()


#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, idx):
#         img_path = self.img_paths[idx]
#         seg_path = self.seg_paths[idx]
#         rel_path = self.rel_img_paths[idx]
#         mask_path = os.path.join(self.mask_dir, rel_path)
#         os.makedirs(os.path.dirname(mask_path), exist_ok=True)
#         # img_path = os.path.join(self.image_dir, img_name)
#         image = Image.open(img_path).convert('RGB')
#         seg_image = Image.open(seg_path).convert("L")


#         seg = (np.array(seg_image) > 0).astype(np.float32) # becare full here
#         # seg = torch.from_numpy(seg).to(self.pipeline.device)
#         seg = torch.from_numpy(seg).type(torch.float16).cuda()
#         if random.random() < self.scheme_ratio:
#             gen_mask, _ = self.mask_gen_a(seg)
#         else:
#             gen_mask, _ = self.mask_gen_b(seg)
#         if self.transform:
#             image = self.transform(image)

#         # Convert to CPU NumPy array
#         gen_mask_np = np.squeeze(gen_mask)

#         # Ensure values are 0 or 255
#         try:
#             gen_mask_img = Image.fromarray((gen_mask_np * 255).astype(np.uint8))
#         except:
#             print(gen_mask_np.shape)
#             print(rel_path)
#             exit(0)
#         gen_mask_img.save(mask_path)
#         return image

# def main():
#     print("Generate mask")
    
#     # Define transformations
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor()
#     ])

#     dataset = MaskGenerateDataset("/home/ubuntu/inpaint_full/", "/home/ubuntu/inpaint_seg/images_segment", "/home/ubuntu/inpaint_full/mask/", "/home/ubuntu/inpaint_full/test.txt", transform=transform)
#     dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
#     for image in dataloader:
#         print(image.shape)

# if __name__ == "__main__":
#     main()