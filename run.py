"""
Description:
    Run inference on an input image. Top level script for framework.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from source.PSG import PSG

def define_and_parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-img", type=str, required=True)
    argparser.add_argument(
        "-ot",
        "--output_type",
        choices=["mesh", "psg", "textured_mesh"],
        default="mesh"
    )
    args = argparser.parse_args()
    return args.img, args.ot

def visualise_points(points):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:,0], points[:,1], points[:,2], s=10)
    ax.set_axis_off()
    plt.show()

def main():
    img = "D:/ShapeNetRendering/ShapeNetRendering/02691156/1a888c2c86248bbcf2b0736dd4d8afe0/rendering/02.png"
    psg = PSG("./models/large_model_1")
    point_set = psg.generate_pointset(img)
    visualise_points(point_set)

if __name__ == "__main__":
    main()