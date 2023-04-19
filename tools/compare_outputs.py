"""
Description:
    Visualise predictions next to each other, ground truth point cloud and
    the input image.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_images(filename: str) -> np.ndarray:
    # Load in images
    file = open(filename, "rb")
    imgs = np.array(pickle.load(file))
    file.close()
    # Normalise images
    imgs = imgs/255.0
    return imgs

def load_point_cloud(filename: str) -> np.ndarray:
    # Load in point clouds
    file = open(filename, "rb")
    point_sets = np.array(pickle.load(file))
    file.close()
    return point_sets

def visualise_data(img: np.ndarray, point_sets: np.ndarray, ps_names: list):
    # Show rendering of textured 3D mesh
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1 + len(point_sets), 1)
    ax.set_title("Rendering")
    plt.imshow(img, cmap = plt.cm.binary)

    # Show point clouds
    for i, points in enumerate(point_sets):
        ax = fig.add_subplot(1, 1 + len(point_sets), 2 + i, projection="3d")
        ax.scatter(points[:,0], points[:,1], points[:,2], s=5, c="red")
        ax.set_axis_on()
        ax.set_aspect("equal")
        ax.set_title(ps_names[i])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_zticklabels([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.view_init(0, 0)

    plt.show()

def main():
    # Find files to be compared
    dirname = sys.argv[1]
    img_file = ""
    pnt_files = []
    for item in os.listdir(dirname):
        if "img" in item:
            img_file = item
        if "pnt" in item:
            pnt_files.append(item)
    # Load in images and point sets
    imgs = load_images(f"{dirname}/{img_file}")
    pnt_sets = []
    ps_names = []
    for pnt_file in pnt_files:
        pnts = load_images(f"{dirname}/{pnt_file}")
        pnt_sets.append(pnts)
        pnt_file = os.path.splitext(pnt_file)[0]
        if "test_pnts" == pnt_file:
            pnt_file = "Ground Truth"
        else:
            pnt_file = pnt_file[:-5]
        ps_names.append(pnt_file)
    # Visualise
    for i, img in enumerate(imgs):
        point_sets = []
        for pnt_set in pnt_sets:
            point_sets.append(pnt_set[i])
        visualise_data(img, point_sets, ps_names)

if __name__ == "__main__":
    main()