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

def main():
    """Load and format Shapenet dataset"""
    shapenet = tfg.datasets.shapenet.Shapenet()
    shapenet.load()

    """Build network according to point set gen architecture"""
    # Evaluate the chamfer distance between 2 point clouds
    chamfer_distance = tfg.nn.loss.chamfer_distance.evaluate()

    """Train network"""
    """Test network"""
    pass

if __name__ == "__main__":
    main()