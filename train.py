"""
Description:
    Script for creating graph, training and testing network.
"""

import json
import pickle
import numpy as np
from keras import layers, models
from tensorflow_graphics.nn import loss

EPOCHS = 10
BATCH_SIZE = 14
POINTS = 256

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

def main():
    """Load Data"""
    images = load_images("data/images.pickle")[0:]
    gt_point_sets = load_point_cloud("data/points.pickle")

    """Create Model"""
    model = models.Sequential()

    # Add Convolutional Layers
    model.add(layers.Conv2D(16, (3,3), activation="relu", input_shape=images.shape[1:]))
    model.add(layers.Conv2D(16, (3,3), activation="relu"))
    model.add(layers.Conv2D(16, (3,3), activation="relu"))

    # Add Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(32))
    model.add(layers.Dense(POINTS*3))

    # Reshape to required output
    model.add(layers.Reshape((POINTS,3)))

    model.summary()

    # Implement chamfer distance loss function and compile model
    loss_function = loss.chamfer_distance.evaluate
    model.compile(
        optimizer="adam",
        loss=loss_function,
        metrics=["accuracy"]
    )

    model.fit(images, gt_point_sets, epochs=EPOCHS, validation_split=0.2)
    model.save("saved_model")

if __name__ == "__main__":
    main()