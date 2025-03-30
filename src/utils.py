import os
import cv2
from typing import Dict
import numpy as np
from PIL import Image, ImageOps

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


class DataProcessorQ1():
    def __init__(self, args):
        self.images_dir = args.q1_images_dir
        self.masks_dir = args.q1_masks_dir

    def __call__(self):
        images_path =  [f for f in os.listdir(os.path.join(dir, 'raw')) if f.endswith(('.jpeg', '.png', '.jpg'))]

    
    


def load_data_q1(dir:str = 'data/q1_folder') -> Dict: # narrow, good, large

    raws_files = [f for f in os.listdir(os.path.join(dir, 'raw')) if f.endswith(('.jpeg', '.png', '.jpg'))]
    raws_files.sort()
    output = {}
    for f in raws_files:
        prefix = f.split('.')[0]
        postfix = f.split('.')[1]
        raw = Image.open(os.path.join(dir, 'raw', f))
        mask = Image.open(os.path.join(dir, 'masks', f'{prefix}_mask.{postfix}')).convert('L')
        inv_mask = ImageOps.invert(mask)

        masked_image =  raw.copy()
        masked_image.putalpha(mask)
        inv_masked_image = raw.copy()
        inv_masked_image.putalpha(inv_mask)
        output[f.split('.')[0][2:]]= {'raw' : raw, 'mask' : mask, 'inv_mask' : inv_mask, 'masked_image': masked_image, 'inv_masked_image': inv_masked_image}
    return output


def canny2mask(gray):
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilated, 
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    
    # largest_contour = max(contours, key=cv2.contourArea) # TODO: Use maximum

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), cv2.FILLED)
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask