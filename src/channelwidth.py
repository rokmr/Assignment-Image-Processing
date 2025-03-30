import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt


class ChannelWidth():
    def __init__(self):
        pass

    def __call__(self, image, mask):
        inv_mask = ImageOps.invert(mask)
        masked_image = image.copy()
        masked_image.putalpha(mask)
        inv_masked_image = image.copy()
        inv_masked_image.putalpha(inv_mask)

        dilated_mask = np.array(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_mask = cv2.dilate(dilated_mask, kernel, iterations=5)

        dilated_masked_image = image.copy()
        dilated_masked_image.putalpha(Image.fromarray(dilated_mask))

        np_dilated_masked_image = np.array(dilated_masked_image)
        canny = cv2.Canny(np_dilated_masked_image[:,:,:3] * np_dilated_masked_image[:,:,3:4],50,150)

        dilated_mask = np.array(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        final_dilated_mask = cv2.dilate(dilated_mask, kernel, iterations=3)
        final_canny_image = (canny * final_dilated_mask) * 255
        new_mask = self.canny2mask(final_canny_image)
        result = self.measure_pipe_width_with_normals(new_mask, boundary_thickness=2, skeleton_point_size=2)
        return result
    
    def canny2mask(self, gray):
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(gray, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]  # Get top 10 largest contours
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, (255), cv2.FILLED)
        mask = cv2.erode(mask, kernel, iterations=1)
        return mask

    def prune_skeleton(self, skeleton, min_branch_length=10):
        labeled_skeleton = label(skeleton)
        for region in regionprops(labeled_skeleton):
            if region.area < min_branch_length:
                for coord in region.coords:
                    skeleton[coord[0], coord[1]] = 0
        return skeleton

    def measure_pipe_width_with_normals(self, img, min_width=47, max_width=74, boundary_thickness=4, skeleton_point_size=3):
        # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        skeleton = skeletonize(binary / 255.0)
        skeleton = (skeleton * 255).astype(np.uint8)
        dist_transform, indices = distance_transform_edt(binary, return_indices=True)
        colored_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        y_coords, x_coords = np.where(skeleton > 0)
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            radius = int(dist_transform[y, x])
            width = radius * 2
            boundary_y, boundary_x = indices[0][y, x], indices[1][y, x]
            dx = boundary_x - x
            dy = boundary_y - y
            norm = np.sqrt(dx**2 + dy**2)

            if norm > 0:
                dx, dy = dx/norm, dy/norm

            if width < min_width or width > max_width:
                color = [255, 0, 0]  
            else:
                color = [0, 255, 0]  

            for factor in [-radius, radius]:
                normal_x = int(x + dx * factor)
                normal_y = int(y + dy * factor)

                if 0 <= normal_x < binary.shape[1] and 0 <= normal_y < binary.shape[0]:
                    cv2.circle(colored_img, (normal_x, normal_y), boundary_thickness, color, -1)

        return colored_img



if __name__ == '__main__':
    image = Image.open('data/q1_folder/raw/1_large.png')
    mask = Image.open('data/q1_folder/masks/1_large_mask.png').convert('L')

    channelwidth = ChannelWidth()

    output = channelwidth(image, mask)

    Image.fromarray(output).save('output.png') 