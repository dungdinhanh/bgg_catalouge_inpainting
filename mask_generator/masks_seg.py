import math
import random
import hashlib
import logging
from enum import Enum

import cv2
import matplotlib.pyplot as plt
import numpy as np
import json

# from powe .utils import LinearRamp

from utils import LinearRamp

import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats

LOGGER = logging.getLogger(__name__)

def calculate_padding(image_size, kernel_size, stride=1):
  """
  Calculates padding values for convolution to maintain input size.

  Args:
      image_size (int): The size of the image dimension (height or width).
      kernel_size (int): The size of the filter kernel.
      stride (int, optional): The stride of the convolution. Defaults to 1.

  Returns:
      tuple: A tuple containing the padding values for top/bottom and left/right sides.
  """
  padding = ((image_size - 1) * stride + kernel_size - image_size) // 2
  return padding[1], padding[1], padding[0], padding[0]


def pad_image_with_filter_size(img, filter_size):
    padding_left, padding_right, padding_top, padding_bottom = calculate_padding(np.asarray(img.shape), filter_size)
    return F.pad(img, [padding_left, padding_right, padding_top, padding_bottom])

def convolve_image_with_box(img, filter_size):
    conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=filter_size, bias=False)
    conv_layer.weight.data = torch.ones(1, 1, filter_size[0], filter_size[1]).type(torch.float16).to(img.device)
    # conv_layer.weight.data = torch.ones(1, 1, filter_size[0], filter_size[1]).type(torch.float16)
    with torch.no_grad():
        img_sum = conv_layer(img[None, None, ...])
    return img_sum

class RandomRectangleMaskFromDistDefectGenerator:
    def __init__(self, annots_file, min_times=0, max_times=3, out_size=512, ramp_kwargs=None):
        self.min_times = min_times
        self.max_times = max_times
        self.out_size = out_size
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

        # Build box distribution from file
        with open(annots_file) as json_data:
            bbox_data = json.load(json_data)
            json_data.close()

        all_boxes = []
        for i in range(len(bbox_data)):
            if len(bbox_data[str(i)].keys()) == 0:
                continue
            height = bbox_data[str(i)]['h']
            width = bbox_data[str(i)]['w']

            bboxes = bbox_data[str(i)]["bboxes"]
            box_w = np.asarray([box_i[2] - box_i[0] for box_i in bboxes])
            box_h = np.asarray([box_i[3] - box_i[1] for box_i in bboxes])

            min_hw = min([height, width])
            factor = out_size / min_hw

            height *= factor
            width *= factor
            box_w = box_w * factor
            box_h = box_h * factor

            box_w = box_w / width
            box_h = box_h / height

            boxes = np.stack([box_w, box_h], axis=0)

            all_boxes.append(boxes)

        all_boxes = np.concatenate(all_boxes, axis=1)
        self.max_size = np.percentile(all_boxes, 95)
        self.min_size = np.percentile(all_boxes, 5)
        self.gkde_obj = stats.gaussian_kde(all_boxes)

    def __call__(self, img, boxes, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_bbox_max_size = self.min_size + (self.max_size - self.min_size)*coef
        cur_max_times = int(self.min_times + (self.max_times - self.min_times) * coef)
        times = np.random.randint(self.min_times, cur_max_times + 1)

        height, width, _ = img.shape
        mask = np.zeros((height, width), np.float32)
        seg = np.zeros((height, width), np.float32)

        # Boxes are [x1, y1, w, h]
        boxes = np.asarray(boxes)
        boxes = np.round(boxes).astype(int)
        boxes[:, 0] = np.clip(boxes[:, 0], 0, width-1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, height-1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, height)

        # Generate existing mask of given boxes. Not used, only for debugging purposes
        for i in range(len(boxes)):
            seg[boxes[i][1]:(boxes[i][1]+boxes[i][3]), boxes[i][0]:(boxes[i][0]+boxes[i][2])] = 1.0

        # VIS
        if 0:
            plt.imsave("/home/ubuntu/xvdb/PycharmProjects/defectcorrection/lama/img.png", img)
            plt.imsave("/home/ubuntu/xvdb/PycharmProjects/defectcorrection/lama/seg.png", seg)


        random_boxes = self.gkde_obj.resample(times, 1337)
        random_boxes = np.clip(random_boxes, self.min_size, cur_bbox_max_size)
        random_boxes[0,:] *= width; random_boxes[1,:] *= height
        random_boxes = random_boxes.astype(int)

        gen_success = []
        for i in range(times):
            box_width = random_boxes[0, i]
            box_height = random_boxes[1, i]

            # Find areas that are viable for generating box
            seg_i = np.zeros((height, width), np.float32)
            # Generate boxes on the mask
            for j in range(len(boxes)):
                # Adjust each box so that it won't allow the generated mask to overlap with the box
                x1 = np.maximum(boxes[j][0] - (box_width + 1), 0)
                y1 = np.maximum(boxes[j][1] - (box_height + 1), 0)
                x2 = boxes[j][0] + boxes[j][2]
                y2 = boxes[j][1] + boxes[j][3]
                seg_i[y1:y2, x1:x2] = 1.0
            seg_bin = seg_i == 0
            # Prevent boxes being generated outside the image region
            seg_bin[:, -(box_width+1):] = False
            seg_bin[-(box_height + 1):, :] = False

            # VIS
            if 0:
                seg_vis = seg_i.copy()
                seg_i_vis = seg_bin.copy()
                plt.imsave("/home/ubuntu/xvdb/PycharmProjects/defectcorrection/lama/seg_i.png", seg_vis)
                plt.imsave("/home/ubuntu/xvdb/PycharmProjects/defectcorrection/lama/seg_bin.png", seg_i_vis)

            if np.all(~seg_bin):
                gen_success.append(False)
                continue
            else:
                gen_success.append(True)
                locy, locx = np.where(seg_bin)
                pix_idx = np.random.randint(0, high=len(locy), size=1)[0]
                start_y = locy[pix_idx]
                start_x = locx[pix_idx]
                mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1

                # VIS
                if 0:
                    plt.imsave("/home/ubuntu/xvdb/PycharmProjects/defectcorrection/lama/mask_i.png", mask)

        # VIS
        if 0:
            plt.imsave("/home/ubuntu/xvdb/PycharmProjects/defectcorrection/lama/mask.png", mask)

        gen_success = np.asarray(gen_success)
        success = np.any(gen_success)
        return mask[None, ...], success

class RandomRectangleMaskFromDistGenerator:
    def __init__(self, annots_file, min_times=0, max_times=3, out_size=512, ramp_kwargs=None):
        self.min_times = min_times
        self.max_times = max_times
        self.out_size = out_size
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

        # Build box distribution from file
        with open(annots_file) as json_data:
            bbox_data = json.load(json_data)
            json_data.close()

        all_boxes = []
        for i in range(len(bbox_data)):
            if len(bbox_data[str(i)].keys()) == 0:
                continue
            height = bbox_data[str(i)]['h']
            width = bbox_data[str(i)]['w']

            bboxes = bbox_data[str(i)]["bboxes"]
            box_w = np.asarray([box_i[2] - box_i[0] for box_i in bboxes])
            box_h = np.asarray([box_i[3] - box_i[1] for box_i in bboxes])

            min_hw = min([height, width])
            factor = out_size / min_hw

            height *= factor
            width *= factor
            box_w = box_w * factor
            box_h = box_h * factor

            box_w = box_w / width
            box_h = box_h / height

            boxes = np.stack([box_w, box_h], axis=0)

            all_boxes.append(boxes)

        all_boxes = np.concatenate(all_boxes, axis=1)
        self.max_size = np.percentile(all_boxes, 95)
        self.min_size = np.percentile(all_boxes, 5)
        self.gkde_obj = stats.gaussian_kde(all_boxes)

    def __call__(self, img, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_bbox_max_size = self.min_size + (self.max_size - self.min_size)*coef
        cur_max_times = int(self.min_times + (self.max_times - self.min_times) * coef)
        times = np.random.randint(self.min_times, cur_max_times + 1)

        height, width, _ = img.shape
        mask = np.zeros((height, width), np.float32)

        random_boxes = self.gkde_obj.resample(times, 1337)
        random_boxes = np.clip(random_boxes, self.min_size, cur_bbox_max_size)
        random_boxes[0,:] *= width; random_boxes[1,:] *= height
        random_boxes = random_boxes.astype(int)

        for i in range(times):
            x_loc = np.random.randint(0, high=width - random_boxes[0, i], size=1)[0]
            y_loc = np.random.randint(0, high=height - random_boxes[1, i], size=1)[0]
            mask[y_loc:(y_loc + random_boxes[1, i]), x_loc:(x_loc + random_boxes[0, i])] = 1

        # VIS
        if 0:
            plt.imsave("mask_i.png", mask)

        return mask[None, ...], True

class RandomRectangleMaskWithSegmGenerator:
    def __init__(self, margin=5, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3, ramp_kwargs=None):
        self.margin = margin
        self.bbox_min_size = bbox_min_size
        self.bbox_max_size = bbox_max_size
        self.min_times = min_times
        self.max_times = max_times
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, seg, iter_i=None, raw_image=None, device="cpu"):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_bbox_max_size = int(self.bbox_min_size + 1 + (self.bbox_max_size - self.bbox_min_size) * coef)
        cur_max_times = int(self.min_times + (self.max_times - self.min_times) * coef)

        height, width = seg.shape
        mask = np.zeros((height, width), np.float32)
        bbox_max_size = min(cur_bbox_max_size, height - self.margin * 2, width - self.margin * 2)
        if bbox_max_size <= 0:
            if height <= self.margin*2:
                h_margin = 0
            else:
                h_margin = self.margin
            if width <= self.margin * 2:
                w_margin = 0
            else:
                w_margin = self.margin
            bbox_max_size = min(cur_bbox_max_size, height - h_margin * 2, width - w_margin * 2)
        else:
            h_margin = self.margin
            w_margin = self.margin

        times = np.random.randint(self.min_times, cur_max_times + 1)
        gen_success = []
        # seg.to(device)
        if self.bbox_min_size >= bbox_max_size:
            if bbox_max_size < 10:
                print("Weird, please check again the path")
                print(raw_image)

            bbox_min_size = bbox_max_size/2
            print(f"bbox min size: {bbox_min_size}")
            print(f"bbox max size: {bbox_max_size}")
            
        else:
            bbox_min_size = self.bbox_min_size
        if bbox_min_size >= bbox_max_size:
            print(raw_image)
            print(f"min: {bbox_min_size}")
            print(f"max: {bbox_max_size}")
        for i in range(times):
            box_width = np.random.randint(bbox_min_size, bbox_max_size)
            box_height = np.random.randint(bbox_min_size, bbox_max_size)


            # Perform convolution to figure out viable background locations for placing masks
            filter_width = box_width
            filter_height = box_height
            if filter_width % 2 == 0:
                filter_width += 1
            if filter_height % 2 == 0:
                filter_height += 1
            seg_pad = pad_image_with_filter_size(seg, np.asarray([filter_height, filter_width]))
            sum_seg = convolve_image_with_box(seg_pad, [filter_height, filter_width])
            sum_seg_bin = sum_seg.squeeze().cpu().numpy() >= (filter_height*filter_width - 1)

            # VIS
            if 0:
                seg_vis = seg.cpu().numpy()
                sum_seg_vis = sum_seg.squeeze().cpu().numpy()
                sum_seg_bin_vis = sum_seg_bin
                plt.imsave("seg.png", seg_vis)
                plt.imsave("seg_sum.png", sum_seg_vis)
                plt.imsave("sum_seg_bin.png", sum_seg_bin_vis)

            if np.all(~sum_seg_bin):
                gen_success.append(False)
                continue
            else:
                gen_success.append(True)
                locy, locx = np.where(sum_seg_bin)
                pix_idx = np.random.randint(0, high=len(locy), size=1)[0]
                start_y = locy[pix_idx] - int(box_height/2)
                start_x = locx[pix_idx] - int(box_width/2)
                mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1

                # VIS
                if 0:
                    plt.imsave("mask_i.png", mask)

        # If couldn't generate a single mask, generate one in a random location
        if len(gen_success) == 0:
            gen_success.append(False)
        if np.all(~np.asarray(gen_success)):
            box_width = np.random.randint(bbox_min_size, bbox_max_size)
            box_height = np.random.randint(bbox_min_size, bbox_max_size)
            start_x = np.random.randint(w_margin, width - w_margin - box_width + 1)
            start_y = np.random.randint(h_margin, height - h_margin - box_height + 1)
            mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1

        return mask[None, ...], np.any(np.asarray(gen_success))

class RandomRectangleMaskWithSegmOverlapGenerator:
    def __init__(self, margin=5, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3, overlap_min=0.3, overlap_max=0.6, ramp_kwargs=None):
        self.margin = margin
        self.bbox_min_size = bbox_min_size
        self.bbox_max_size = bbox_max_size
        self.min_times = min_times
        self.max_times = max_times
        self.overlap_min = overlap_min
        self.overlap_max = overlap_max
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, seg, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_bbox_max_size = int(self.bbox_min_size + 1 + (self.bbox_max_size - self.bbox_min_size) * coef)
        cur_max_times = int(self.min_times + (self.max_times - self.min_times) * coef)

        height, width = seg.shape
        mask = np.zeros((height, width), np.float32)
        bbox_max_size = min(cur_bbox_max_size, height - self.margin * 2, width - self.margin * 2)
        if bbox_max_size <= 0:
            if height <= self.margin*2:
                h_margin = 0
            else:
                h_margin = self.margin
            if width <= self.margin * 2:
                w_margin = 0
            else:
                w_margin = self.margin 
            bbox_max_size = min(cur_bbox_max_size, height - h_margin * 2, width - w_margin * 2)
        else:
            h_margin = self.margin
            w_margin = self.margin

        seg = 1.0 - seg # Flip background and foreground

        times = np.random.randint(self.min_times, cur_max_times + 1)
        gen_success = []
        # print(times)
        # make sure bbox_min_size smaller than bbox_max_size
        if self.bbox_min_size >= bbox_max_size:
            if bbox_max_size < 10:
                print("Weird, please check again the path")
                print(raw_image)
                

            bbox_min_size = bbox_max_size/2
            print(f"bbox min size: {bbox_min_size}")
            print(f"bbox max size: {bbox_max_size}")
            
        else:
            bbox_min_size = self.bbox_min_size
        if bbox_min_size >= bbox_max_size:
            print(raw_image)
            print(f"min: {bbox_min_size}")
            print(f"max: {bbox_max_size}")
        for i in range(times):
            
            box_width = np.random.randint(bbox_min_size, bbox_max_size)
            box_height = np.random.randint(bbox_min_size, bbox_max_size)

            # Perform convolution to figure out viable background locations for placing masks
            filter_width = box_width
            filter_height = box_height
            if filter_width % 2 == 0:
                filter_width += 1
            if filter_height % 2 == 0:
                filter_height += 1

            # Performing convolution and then padding to account for
            sum_seg_need_pad = convolve_image_with_box(seg, [filter_height, filter_width])
            sum_seg = pad_image_with_filter_size(sum_seg_need_pad.squeeze(), np.asarray([filter_height, filter_width]))
            max_sum = filter_height*filter_width - 1
            coverage_min = int(max_sum*self.overlap_min)
            coverage_max = int(max_sum*self.overlap_max)
            sum_seg_bin = np.bitwise_and(sum_seg.squeeze().cpu().numpy() >= coverage_min, sum_seg.squeeze().cpu().numpy() < coverage_max)

            # VIS
            if 0:
                seg_vis = seg.cpu().numpy()
                sum_seg_vis = sum_seg.squeeze().cpu().numpy()
                sum_seg_bin_vis = sum_seg_bin
                plt.imsave("seg_flip.png", seg_vis)
                plt.imsave("seg_sum.png", sum_seg_vis)
                plt.imsave("sum_seg_bin.png", sum_seg_bin_vis)

            if np.all(~sum_seg_bin):
                gen_success.append(False)
                continue
            else:
                gen_success.append(True)
                locy, locx = np.where(sum_seg_bin)
                pix_idx = np.random.randint(0, high=len(locy), size=1)[0]
                start_y = locy[pix_idx] - int(box_height/2)
                start_x = locx[pix_idx] - int(box_width/2)
                mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1

                # VIS
                if 0:
                    plt.imsave("mask_i.png", mask)

        # If couldn't generate a single mask, generate one in a random location
        if len(gen_success) == 0:
            gen_success.append(False)
        if np.all(~np.asarray(gen_success)):
            box_width = np.random.randint(bbox_min_size, bbox_max_size)
            box_height = np.random.randint(bbox_min_size, bbox_max_size)
            start_x = np.random.randint(w_margin, width - w_margin - box_width + 1)
            start_y = np.random.randint(h_margin, height - h_margin - box_height + 1)
            mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1

        return mask[None, ...], np.any(np.asarray(gen_success))


class RandomRectangleMaskGenerator:
    def __init__(self, margin=5, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3):
        self.margin = margin
        self.bbox_min_size = bbox_min_size
        self.bbox_max_size = bbox_max_size
        self.min_times = min_times
        self.max_times = max_times

    def __call__(self, image, raw_image):
        if image.ndim == 3:
            height, width = image.shape[:2]
        elif image.ndim == 2:
            height, width = image.shape
        else:
            raise ValueError("Input image must be 2D (grayscale) or 3D (color) NumPy array.")

        mask = np.zeros((height, width), dtype=np.float32)

        max_size_y = height - 2 * self.margin
        max_size_x = width - 2 * self.margin
        safe_max_size = max(1, min(self.bbox_max_size, max_size_y, max_size_x))

        safe_min_size = min(self.bbox_min_size, safe_max_size)
        if safe_min_size < 1:
            safe_min_size = 1

        times = np.random.randint(self.min_times, self.max_times + 1)

        for _ in range(times):
            if safe_min_size >= safe_max_size:
                box_width = box_height = safe_max_size
            else:
                box_width = np.random.randint(safe_min_size, safe_max_size + 1)
                box_height = np.random.randint(safe_min_size, safe_max_size + 1)

            range_x = width - 2 * self.margin - box_width
            range_y = height - 2 * self.margin - box_height
            if range_x < 0 or range_y < 0:
                continue  # Box doesn't fit

            start_x = np.random.randint(self.margin, self.margin + range_x + 1)
            start_y = np.random.randint(self.margin, self.margin + range_y + 1)

            mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1.0

        return mask[None, ...]  # Add channel dimension