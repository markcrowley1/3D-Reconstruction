"""
Description:
    Script supports visualisation of training data pairs - images with
    their corresponding point clouds that should be properly aligned.
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def define_and_parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-img", type=str, required=True)
    argparser.add_argument("-pnt", type=str, required=True)
    args = argparser.parse_args()
    parsed_args = [args.img, args.pnt]
    return parsed_args

def visualise_data(img: np.ndarray, points: np.ndarray):
    # Show rendering of textured 3D mesh
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Rendering")
    plt.imshow(img, cmap = plt.cm.binary)

    # Show corresponding ground truth 3d point cloud
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(points[:,0], points[:,1], points[:,2], s=10, c="red")
    ax.set_axis_on()
    ax.set_aspect("equal")
    ax.set_title("Point Cloud")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_zticklabels([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.view_init(0, 0)

    plt.show()

def main():
    # Read in images and corresponding point sets
    paths = define_and_parse_args()

    file = open(paths[0], "rb")
    imgs = np.array(pickle.load(file))
    imgs = imgs/255.0
    file.close()

    file = open(paths[1], "rb")
    point_sets = np.array(pickle.load(file))
    file.close()

    print(len(imgs), len(point_sets))
    for i in range(len(imgs)):
        visualise_data(imgs[i], point_sets[i])

if __name__ == "__main__":
    main()