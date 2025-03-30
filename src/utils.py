import os
import cv2
from typing import Dict
import numpy as np
from PIL import Image, ImageOps

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)