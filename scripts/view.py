"""
Description:
    Quick viewing test on shapenet models
"""

import numpy as np
import trimesh
from trimesh.sample import sample_surface
import open3d as o3d
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_graphics.datasets.shapenet import Shapenet
import tensorflow_graphics as tfg

from matplotlib import pyplot as plt

def main():
    """Download random model and plot using trimesh"""
    # Paths to shapenet models
    obj_file = "D:/shapenet_base/shapenet_core/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj"
    obj_file_2 = "D:/shapenet_base/shapenet_core/02691156/1a6ad7a24bb89733f412783097373bdc/models/model_normalized.obj"
    # Scene has texture but cannot be sampled - mesh can be sampled
    scene = trimesh.load(obj_file_2)
    mesh = trimesh.load(obj_file_2, force="mesh")
    mesh.show()

    # Create ground truth point cloud
    samples = sample_surface(mesh, 2048)
    points = samples[0]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_axis_off()
    plt.show()

    """Code to plot obj file using open3d"""
    # mesh = o3d.io.read_triangle_mesh(obj_file)
    # o3d.visualization.draw_plotly([mesh],
    #                             up=[0, 1, 0],
    #                             front=[0, 0, 1],
    #                             lookat=[0.0, 0.1, 0.0],
    #                             zoom=0.5)


if __name__ == "__main__":
    main()