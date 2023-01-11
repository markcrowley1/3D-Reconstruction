"""
Description:
    Initial script for training and experimenting with point set gen based network
"""
import trimesh
import keras
import tensorflow as tf
import tensorflow_graphics as tfg
import cv2
import pandas as pd
from tensorflow_graphics.io.triangle_mesh import load
from tensorflow_graphics.geometry.representation.mesh.sampler import area_weighted_random_sample_triangle_mesh
from matplotlib import pyplot as plt

def visualise_points(points):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:,0], points[:,1], points[:,2], s=100)
    ax.set_axis_off()
    plt.show()

def main():
    """Load and format Shapenet dataset"""
    model_path = "D:/shapenet_base/shapenet_core/02691156/1a6ad7a24bb89733f412783097373bdc/models/model_normalized.obj"
    # Tensorflow graphics
    tf_mesh: trimesh.Trimesh = load(model_path, "obj", force="mesh")
    sampled_points, sampled_face_indices = area_weighted_random_sample_triangle_mesh(tf_mesh.vertices, tf_mesh.faces, 1024)
    points = sampled_points.numpy()
    visualise_points(points)

    """Build network according to point set gen architecture"""
    # # Evaluate the chamfer distance between 2 point clouds
    # chamfer_distance = tfg.nn.loss.chamfer_distance.evaluate()

    """Train network"""
    """Test network"""
    pass

if __name__ == "__main__":
    main()