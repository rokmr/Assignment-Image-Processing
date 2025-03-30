import os
import cv2
import numpy as np
import matplotlib 
from PIL import Image
import json

class TemplateMatching():
    def __init__(self, config):
        self.config = config

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
