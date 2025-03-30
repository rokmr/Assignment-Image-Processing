import os
from tqdm import tqdm
import json
import argparse
import numpy as np
from PIL import Image

from src import TemplateMatching, ChannelWidth


parser = argparse.ArgumentParser(description='Assignment')
parser.add_argument('--q1', action='store_true', help='Channel Width (Q1)')
parser.add_argument('--q1_images_dir',type=str, default='data/q1_folder/raw' )
parser.add_argument('--q1_masks_dir',type=str, default='data/q1_folder/masks' )
parser.add_argument('--q1_output_dir', type=str, default='results/q1')
parser.add_argument('--q2', action='store_true', help='Run template matching (Q2)')
parser.add_argument('--q2_images_dir', type=str, default='data/q2_folder/targets')
parser.add_argument('--q2_template_dir', type=str, default='data/q2_folder/templates') 
parser.add_argument('--q2_output_dir', type=str, default='results/q2')
parser.add_argument('--q2_config', type=str, default='config/tm_ccoeff_normed.json')
args = parser.parse_args()

def q1_solver(args):
    os.makedirs(args.q1_output_dir, exist_ok=True)
    channelwidth = ChannelWidth()
    images_name = [f for f in os.listdir(args.q1_images_dir) if f.endswith(('.jpeg', '.png', '.jpg', '.bmp'))]
    # masks_name = [f for f in os.listdir(args.q1_masks_dir) if f.endswith(('.jpeg', '.png', '.jpg', '.bmp'))]


    if len(images_name)==0:
        raise ValueError('Please provide correct directory')
    
    for image_name in tqdm(images_name):
        # for mask_name in masks_name:
        mask_name = image_name.split('.')[0]+ '_mask.'+image_name.split('.')[1]
        image = Image.open(os.path.join(args.q1_images_dir, image_name))
        mask = Image.open(os.path.join(args.q1_masks_dir, mask_name)).convert('L')
        output = channelwidth(image, mask)
        Image.fromarray(output).save(os.path.join(args.q1_output_dir, mask_name))

def q2_solver(args):
    os.makedirs(args.q2_output_dir, exist_ok=True)
    with open(args.q2_config, 'r') as f:
        config = json.load(f)
    templatematcing = TemplateMatching(config)

    images_name = [f for f in os.listdir(args.q2_images_dir) if f.endswith(('.jpeg', '.png', '.jpg', '.bmp'))]
    templates_name = [f for f in os.listdir(args.q2_template_dir) if f.endswith(('.jpeg', '.png', '.jpg', '.bmp'))]

    if len(images_name)==0 or len(templates_name)==0:
        raise ValueError('Please provide correct directory')
        


    for image_name in tqdm(images_name):
        for template_name in templates_name:
            image = np.array(Image.open(os.path.join(args.q2_images_dir, image_name)).convert('L'))
            template = np.array(Image.open(os.path.join(args.q2_template_dir, template_name)).convert('L'))
            output = templatematcing(image, template)
            output_dir = os.path.join(args.q2_output_dir, image_name.split('.')[0])
            os.makedirs(output_dir, exist_ok=True)
            Image.fromarray(output).save(os.path.join(output_dir, template_name))


if __name__ == '__main__':

    if args.q2:
        q2_solver(args)
    if args.q1:
        q1_solver(args)
    print('Done')