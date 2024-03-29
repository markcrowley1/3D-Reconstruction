"""
Description:
    Script for creating graph, training and testing network.
"""

import os
import pickle
import argparse
import datetime
import numpy as np
from keras import layers, models
from tensorflow_graphics.nn import loss
from settings import *

def define_and_parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data", type=str, default="data")
    argparser.add_argument("--test_output", type=str, default="output/nn_psg_output.pickle")
    args = argparser.parse_args()
    imgs = [f"{args.data}/train_imgs.pickle", f"{args.data}/test_imgs.pickle"]
    pnts = [f"{args.data}/train_pnts.pickle", f"{args.data}/test_pnts.pickle"]
    output = args.test_output
    return imgs, pnts, output

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

def gen_model_name() -> str:
    if not os.path.isdir("models"):
        os.mkdir("models")
    default = "models/cnn_"
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return default + date_time

def main():
    """Load Data"""
    imgs, pnts, out_file = define_and_parse_args()
    train_imgs_file, test_imgs_file = imgs
    train_pnts_file, test_pnts_file = pnts

    train_imgs = load_images(train_imgs_file)
    train_pnts = load_point_cloud(train_pnts_file)

    test_imgs = load_images(test_imgs_file)
    test_pnts = load_point_cloud(test_pnts_file)
    print(train_pnts.shape)

    """Create Model"""
    model = models.Sequential()

    # Add Convolutional Layers
    model.add(layers.Conv2D(16, (3,3), activation="relu", input_shape=train_imgs.shape[1:]))
    model.add(layers.Conv2D(16, (3,3), activation="relu"))
    model.add(layers.Conv2D(16, (3,3), activation="relu"))
    
    model.add(layers.Conv2D(32, (3,3), activation="relu"))
    model.add(layers.Conv2D(32, (3,3), activation="relu"))
    model.add(layers.Conv2D(32, (3,3), activation="relu"))

    model.add(layers.Conv2D(64, (3,3), activation="relu"))
    model.add(layers.Conv2D(64, (3,3), activation="relu"))
    model.add(layers.Conv2D(64, (3,3), activation="relu"))

    # Add Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    # model.add(layers.Dropout(0.2))
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

    # Train the model
    history = model.fit(train_imgs, train_pnts, epochs=EPOCHS,
                        validation_split=0.1, callbacks=None)
    
    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(test_imgs, test_pnts)
    print(val_loss, val_accuracy)
    
    # Generate a set of outputs for the model
    output = model.predict(test_imgs)
    if not os.path.isdir("./output"):
        os.mkdir("output")
    pickle_out = open(out_file, "wb")
    pickle.dump(output, pickle_out)
    pickle_out.close()

    # Save the model
    name = gen_model_name()
    model.save(name)
    with open(f'{name}_history.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


if __name__ == "__main__":
    main()