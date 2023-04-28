"""
Description:
    This script evaluates the view synthesis model on a set of input
    and target output data.
"""
import pickle
import argparse
import numpy as np
from keras import models
from keras import backend as K

def define_and_parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", "--test_dir", type=str, required=True)
    argparser.add_argument("-o", "--output", type=str, default= "./test_output")
    args = argparser.parse_args()
    parsed_args = [args.test_dir, args.output]
    return parsed_args

def load_data(filename: str):
        """Load imgs or point sets from pickle file"""
        file = open(filename, "rb")
        data = np.array(pickle.load(file))
        file.close()
        return data

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
    test_dir, output = define_and_parse_args()
    model_name = "./models/view_synthesis_model"
    x = load_data(f"{test_dir}/x_imgs.pickle")/255.0
    t = load_data(f"{test_dir}/t_vals.pickle")
    y = load_data(f"{test_dir}/y_imgs.pickle")/255.0

    model: models.Model = models.load_model(
            model_name,
            custom_objects={'squared_euclidean_distance': squared_euclidean_distance}
        )
    test_loss = model.evaluate([x, t], y)
    print(test_loss)

if __name__ == "__main__":
    main()