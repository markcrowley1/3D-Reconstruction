"""
Description:
    Initial script for training and experimenting with point set gen based network
"""
import trimesh
import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg
from tensorflow_graphics.io.triangle_mesh import load
from tensorflow_graphics.geometry.representation.mesh.sampler import area_weighted_random_sample_triangle_mesh
from matplotlib import pyplot as plt
from keras import utils, layers

IMG_HEIGHT = 512
IMG_WIDTH = 512

def load_images(data_dir: str):
    # training_data = utils.image_dataset_from_directory(data_dir, batch_size=32,image_size=(IMG_HEIGHT, IMG_WIDTH))
    # print(training_data)
    dataset = utils.image_dataset_from_directory(
    data_dir, batch_size=32, image_size=(IMG_HEIGHT, IMG_WIDTH))

    # For demonstration, iterate over the batches yielded by the dataset.
    for data, labels in dataset:
        print(data.shape)  # (64,)
        print(data.dtype)  # string
        print(labels.shape)  # (64,)
        print(labels.dtype)  # int32

    # Example image data, with values in the [0, 255] range
    training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(training_data)

    normalized_data = normalizer(training_data)
    print("var: %.4f" % np.var(normalized_data))
    print("mean: %.4f" % np.mean(normalized_data))

    return

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