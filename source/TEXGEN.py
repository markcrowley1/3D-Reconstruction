"""
Description:
    Class handles texture generation from input image.
"""

import cv2
import numpy as np
from math import floor

class TEXGEN:
    def __init__(self):
        pass

    def read_img(self, img: str) -> np.ndarray:
        """Read image using opencv and return as numpy array (RGB format)"""
        img_array = cv2.imread(img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_array = np.array(img_array)
        return img_array
    
    def generate_texture(self, img: np.ndarray) -> np.ndarray:
        """Generate texture from img stored in np array"""
        # Sample central pixels and split RGB channels to seperate arrays
        height, width, channels = img.shape
        centre_h, centre_w = floor(height/2), floor(width/2)
        sample = img[centre_h-5:centre_h+5, centre_w-5:centre_w+5]
        r, g, b = sample[:,:,0], sample[:,:,1], sample[:,:,2]
        # Get avg value of each channel - create texture with avg colour
        avg_r, avg_g, avg_b = r.mean(), g.mean(), b.mean()
        tex_r = np.ones((255,255))*avg_r
        tex_g = np.ones((255,255))*avg_g
        tex_b = np.ones((255,255))*avg_b
        tex = np.dstack((tex_b, tex_g, tex_r))
        return tex
    
    def save_img(self, texture: np.ndarray, outfile: str):
        cv2.imwrite(outfile, texture)
        return