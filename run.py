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