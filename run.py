"""
Description:
    Run inference on set of images loaded from pickle file.
"""

import os
import sys
import pickle
import argparse
import numpy as np
from keras import models
from tensorflow_graphics.nn import loss

def define_and_parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m", "--model", type=str, required=True)
    argparser.add_argument("-img", "--images", type=str, default="data/test_imgs.pickle")
    argparser.add_argument("-o", "--output", type=str, default="output/nn_output.pickle")
    args = argparser.parse_args()
    parsed_args = [args.model, args.images, args.output]
    return parsed_args

def load_images(filename: str) -> np.ndarray:
    # Load in images
    file = open(filename, "rb")
    imgs = np.array(pickle.load(file))
    file.close()
    # Normalise images
    imgs = imgs/255.0
    return imgs

def main():
    args = define_and_parse_args()
    model_name, img_file, out_file = args
    images = load_images(img_file)
    
    model: models.Sequential = models.load_model(
        model_name,
        custom_objects={'evaluate': loss.chamfer_distance.evaluate}
    )
    output = model.predict(images)

    if not os.path.isdir("./output"):
        os.mkdir("output")
    pickle_out = open(out_file, "wb")
    pickle.dump(output, pickle_out)
    pickle_out.close()

if __name__ == "__main__":
    main()