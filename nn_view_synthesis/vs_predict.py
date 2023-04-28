
"""
Description:
    This script uses the view synthesis model to generate images of an
    object from a new viewpoint.
"""

import pickle
import argparse
import numpy as np
from keras import models
from keras import backend as K

def define_and_parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", "--test_dir", type=str, required=True)
    argparser.add_argument("-o", "--output", type=str, default= "./vs_prediction.pickle")
    args = argparser.parse_args()
    parsed_args = [args.test_dir, args.output]
    return parsed_args

def load_data(filename: str):
        """Load imgs or point sets from pickle file"""
        file = open(filename, "rb")
        data = np.array(pickle.load(file))
        file.close()
        return data

def save_predictions(imgs: np.ndarray, out_file: str):
    """Save predicted images"""
    # Denormalise the predicted images
    imgs = imgs * 255
    # Save
    pickle_out = open(out_file, "wb")
    pickle.dump(imgs, pickle_out)
    pickle_out.close()

def squared_euclidean_distance(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sum(K.square(y_pred - y_true), axis=-1)

def main():
    test_dir, out_file = define_and_parse_args()
    model_name = "./models/view_synthesis_model"
    x = load_data(f"{test_dir}/x_imgs.pickle")/255.0
    t = load_data(f"{test_dir}/t_vals.pickle")
    y = load_data(f"{test_dir}/y_imgs.pickle")/255.0

    model: models.Model = models.load_model(
            model_name,
            custom_objects={'squared_euclidean_distance': squared_euclidean_distance}
        )
    imgs = model.predict([x, t])
    save_predictions(imgs, out_file)

if __name__ == "__main__":
    main()