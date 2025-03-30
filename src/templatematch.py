import os
import cv2
import numpy as np
import matplotlib 
from PIL import Image
import json

class TemplateMatching():
    def __init__(self, config):
        self.config = config

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    
    def scale_image(self, image, percent, maxwh):
        max_width = maxwh[1]
        max_height = maxwh[0]
        max_percent_width = max_width / image.shape[1] * 100
        max_percent_height = max_height / image.shape[0] * 100
        max_percent = 0
        if max_percent_width < max_percent_height:
            max_percent = max_percent_width
        else:
            max_percent = max_percent_height
        if percent > max_percent:
            percent = max_percent
        width = int(image.shape[1] * percent / 100)
        height = int(image.shape[0] * percent / 100)
        result = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
        return result, percent
    
    def __call__(self, image, template):
        pass

    def plot(self):
        pass

    def preprocess(self):
        pass


if __name__ == '__main__':
    image = np.array(Image.open('data/q2_folder/targets/Image_2243.bmp'))
    template = np.array(Image.open('data/q2_folder/targets/Image_2243.bmp'))

    with open('config/tm_ccoeff_normed.json', 'r') as f:
        config = json.load(f)

    templatematcing = TemplateMatching(config)
    output = templatematcing(image, template)
    print('Done')
