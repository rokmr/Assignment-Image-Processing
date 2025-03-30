import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils import Config

class TemplateMatching():
    def __init__(self, config):
        self.image = None
        self.template = None
        self.config = Config(config) if isinstance(config, dict) else config

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
        self.image = image
        self.template = template
        image_maxwh = image.shape
        height, width = template.shape
        all_points = []
        for next_angle in range(self.config.rot_range[0], self.config.rot_range[1], self.config.rot_interval):
            for next_scale in range(self.config.scale_range[0], self.config.scale_range[1], self.config.scale_interval):
                scaled_template_gray, actual_scale = self.scale_image(template, next_scale, image_maxwh)
                if next_angle == 0:
                    rotated_template = scaled_template_gray
                else:
                    rotated_template = self.rotate_image(scaled_template_gray, next_angle)

                matched_points = cv2.matchTemplate(image,rotated_template,cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                if max_val >= self.config.matched_thresh:
                    all_points.append([max_loc, next_angle, actual_scale, max_val])
                

        all_points = sorted(all_points, key=lambda x: -x[3])
        lone_points_list = []
        visited_points_list = []
        for point_info in all_points:
            point = point_info[0]
            scale = point_info[2]
            all_visited_points_not_close = True
            if len(visited_points_list) != 0:
                for visited_point in visited_points_list:
                    if ((abs(visited_point[0] - point[0]) < (width * scale / 100)) and (abs(visited_point[1] - point[1]) < (height * scale / 100))):
                        all_visited_points_not_close = False
                if all_visited_points_not_close == True:
                    lone_points_list.append(point_info)
                    visited_points_list.append(point)
            else:
                lone_points_list.append(point_info)
                visited_points_list.append(point)
        return self.plot(lone_points_list)


    def plot(self, points_list):
        output_image = self.image.copy()
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB)
        
        height, width = self.template.shape
        for point_info in points_list:
            point, angle, scale = point_info[0:3]
            print(f"matched point: {point}, angle: {angle}, scale: {scale}")
            
            rect_width = int(width * scale / 100)
            rect_height = int(height * scale / 100)
            
            center_x = int(point[0] + rect_width/2)
            center_y = int(point[1] + rect_height/2)
            
            box = cv2.boxPoints(((point[0] + rect_width/2, point[1] + rect_height/2), 
                               (rect_width, rect_height), 
                               angle))
            box = np.int32(box)
            
            cv2.drawContours(output_image, [box], 0, (0, 0, 255), 2)  # Red rectangle
            cv2.circle(output_image, (center_x, center_y), 3, (0, 255, 0), -1)  # Green center point
        return output_image

    def preprocess(self):
        pass


if __name__ == '__main__':
    image = np.array(Image.open('data/q2_folder/targets/Image_2243.bmp').convert('L'))
    template = np.array(Image.open('data/q2_folder/templates/h1.png').convert('L'))

    with open('config/tm_ccoeff_normed.json', 'r') as f:
        config = json.load(f)

    templatematcing = TemplateMatching(config)
    output = templatematcing(image, template)
    cv2.imwrite('output.png', output) 
    print('Done')
