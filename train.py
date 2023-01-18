"""
Description:
    Script for creating graph, training and testing network.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg
from tensorflow_graphics.nn import loss
from keras import layers, models, utils, preprocessing

EPOCHS = 28
BATCH_SIZE = 14

IMG_HEIGHT = 512
IMG_WIDTH = 512
OUTPUT_POINTS = 1024
GT_POINTS = OUTPUT_POINTS*10

def np_images(data_dir: str):
    return

def load_images(data_dir: str):
    dataset = utils.image_dataset_from_directory(
    data_dir, batch_size=1, image_size=(IMG_HEIGHT, IMG_WIDTH), labels=None)

    
    size = (256, 256)
    dataset = dataset.map(lambda img: tf.image.resize(dataset, size))

    normalization_layer = layers.Rescaling(1./255)
    normalized_ds = dataset.map(lambda x: (normalization_layer(x)))
    image_batch = next(iter(normalized_ds))

    return image_batch

def load_point_cloud(data_dir: str) -> np.ndarray:
    filename = "points.pickle"
    pc_path = os.path.join(data_dir, filename)
    with open(pc_path, "rb") as file:
        points = pickle.load(file)
    file.close()

    return points

def build_model():
    # Build graph
    model = models.Sequential()

    # Add convolution layers
    model.add(layers.Conv2D(16, 3, activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(layers.Conv2D(16, 3, activation="relu"))
    model.add(layers.Conv2D(16, 3, activation="relu"))

    # Add fully connected layer
    model.add(layers.Flatten())
    model.add(layers.Dense(32))

    # Reshape to give point cloud coordinates
    model.add(layers.Reshape((2,4,4)))

    model.summary()
    
    # Loss function - chamfer distance between estimated and gt point cloud
    loss_function = loss.chamfer_distance.evaluate
    model.compile(
        optimizer="adam",
        loss=loss_function,
        metrics=["accuracy"]
    )

    return model

def main():
    """Load training data"""
    data_dir = "./data/subdir2"
    training_data = load_images(data_dir)
    target_data = load_point_cloud(data_dir)
    print(target_data.shape)

    """Build Model"""
    model: models.Sequential = build_model()
    print(model.output_shape)
    """Train Model"""
    # model.fit(training_data, target_data, epochs=EPOCHS)

    """Test Model"""


    """Save Model"""
    # model.save("Basic_PSG")
    

if __name__ == "__main__":
    main()