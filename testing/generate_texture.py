"""
Description:
    Experimenting with texture generation algorithms.
    First iteration - sample central pixel values and get average. Create
    texture with all pixels of that value.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import floor

def display(img, sample, tex):
    # Show rendering of textured 3D mesh
    fig = plt.figure(figsize=(5, 5))
    fig.add_subplot(1, 3, 1)
    plt.imshow(img)

    # Sample from image
    fig.add_subplot(1, 3, 2)
    plt.imshow(sample)

    # Texture from sample
    fig.add_subplot(1, 3, 3)
    plt.imshow(tex)
    plt.show()

def main():
    img_dir = "D:/ShapeNetRendering/ShapeNetRendering/04256520/1aaee46102b4bab3998b2b87439f61bf/rendering"
    for img in os.listdir(img_dir):
        if img.endswith(".png"):
            # Read image
            img_array = cv2.imread(f"{img_dir}/{img}")
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_array = np.array(img_array)
            print(img_array[0:2])
            # Create texture
            # Sample central pixel values and get average
            height, width, channels = img_array.shape
            centre_h, centre_w = floor(height/2), floor(width/2)
            sample = img_array[centre_h-5:centre_h+5, centre_w-5:centre_w+5]
            r, g, b = sample[:,:,0], sample[:,:,1], sample[:,:,2]
            avg_r, avg_g, avg_b = r.mean(), g.mean(), b.mean()
            tex_r = np.ones((255,255))*avg_r
            tex_g = np.ones((255,255))*avg_g
            tex_b = np.ones((255,255))*avg_b
            tex = np.dstack((tex_b, tex_g, tex_r))
            display(img_array, sample, tex[:,:,::-1]/255)
            cv2.imwrite("texture.png", tex)

if __name__ == "__main__":
    main()