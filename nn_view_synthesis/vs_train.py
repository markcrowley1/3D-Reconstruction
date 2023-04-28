"""
Description:
    Training view synthesis model.
"""

import os
import pickle
import datetime
import argparse
import numpy as np
from keras import layers, models, Input
from keras import backend as K
from vs_settings import *
from vs_batch_fetcher import BatchFetcher, gen_file_list

def define_and_parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data", type=str, default="D:/ShapeNetRendering/texnet_data")
    argparser.add_argument("--test_output", type=str, default="output")
    args = argparser.parse_args()
    return args.data, args.test_output

def load_images(filename: str) -> np.ndarray:
    # Load in images
    file = open(filename, "rb")
    imgs = np.array(pickle.load(file))
    file.close()
    # Normalise images
    imgs = imgs/255.0
    return imgs

def load_transform(filename: str) -> np.ndarray:
    # Load in images
    file = open(filename, "rb")
    transforms = np.array(pickle.load(file))
    file.close()
    return transforms

def squared_euclidean_distance(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sum(K.square(y_pred - y_true), axis=-1)

def gen_model_name() -> str:
    if not os.path.isdir("models"):
        os.mkdir("models")
    default = "models/cnn_"
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return default + date_time


def main():
    # Get filenames for data
    data_dir, out_dir = define_and_parse_args()
    # Get paths to files containing batches. 
    train_files, val_files, test_files = gen_file_list(data_dir)
    print(len(train_files))

    train_generator = BatchFetcher(train_files, BATCH_SIZE)
    val_generator = BatchFetcher(val_files, BATCH_SIZE)

    """Create Model"""
    # Define image and transform inputs
    img_input = Input(shape=(112,112,3), name="Image")
    transform_input = Input(shape=(3), name="Transform")

    # Convolution Branch (Image)
    conv_seq = layers.Conv2D(14, 3, 2, "same", activation="relu")(img_input)
    conv_seq = layers.Conv2D(28, 3, 2, "same", activation="relu")(conv_seq)
    conv_seq = layers.Conv2D(56, 3, 2, "same", activation="relu")(conv_seq)
    conv_seq = layers.Conv2D(112, 3, 2, "same", activation="relu")(conv_seq)
    conv_seq = layers.Flatten()(conv_seq)
    conv_seq = layers.Dense(64)(conv_seq)

    # Fully Connected Branch (Transform Info)
    fc1 = layers.Dense(3)(transform_input)
    fc1 = layers.Dense(32)(fc1)
    fc1 = layers.Dense(32)(fc1)

    # Fully Connected Layer (Combining Branches)
    fc2 = layers.concatenate([conv_seq, fc1])
    fc2 = layers.Dense(512)(fc2)
    fc2 = layers.Dense(1127)(fc2)
    fc2 = layers.Reshape((7,7,23))(fc2)

    deconv_seq = layers.Conv2DTranspose(56, 3, 2, "same", activation="relu")(fc2)
    deconv_seq = layers.Conv2DTranspose(28, 3, 2, "same", activation="relu")(deconv_seq)
    deconv_seq = layers.Conv2DTranspose(14, 3, 2, "same", activation="relu")(deconv_seq)
    outputs = layers.Conv2DTranspose(3, 3, 2, "same", activation="relu")(deconv_seq)

    model = models.Model(
        inputs=[img_input, transform_input],
        outputs=outputs
    )

    model.summary()
    model.compile(
        loss=squared_euclidean_distance,
        optimizer="rmsprop"
    )

    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )

    # Save the model
    name = gen_model_name()
    model.save(name)
    with open(f'{name}_history.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # Make prediction for test images from each category
    if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
    for i, files in enumerate(test_files):
        test_x, test_t, test_y = files
        test_x, test_t = load_images(test_x), load_transform(test_t)
        output = model.predict([test_x, test_t])
        # Save the prediction
        pickle_out = open(f"{out_dir}/nn_output_{i}.pickle", "wb")
        pickle.dump(output * 255, pickle_out)
        pickle_out.close()
        
if __name__ == "__main__":
    main()