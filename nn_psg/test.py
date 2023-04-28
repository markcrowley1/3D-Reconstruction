"""
Description:
    Script for testing a single model or multiple models on a 
    large number of images with gt point sets. Performance metrics are
    generated along with point clouds and the outputs are saved.
"""

import os
import shutil
import pickle
import argparse
import numpy as np
from keras import models
from tensorflow_graphics.nn import loss

def define_and_parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m", "--model", nargs="+", required=True)
    argparser.add_argument("-img", type=str, default="data/test_imgs.pickle")
    argparser.add_argument("-pnt", type=str, default="data/test_pnts.pickle")
    argparser.add_argument("-o", "--output", type=str, default="psg_outputs")
    args = argparser.parse_args()
    parsed_args = [args.model, args.img, args.pnt, args.output]
    return parsed_args

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

def save_predictions(pc: np.ndarray, out_file: str):
    pickle_out = open(out_file, "wb")
    pickle.dump(pc, pickle_out)
    pickle_out.close()

def save_loss(loss_values: list, out_file: str):
    with open(out_file, "w") as f:
        for line in loss_values:
            f.write(f"{line[0]} --- {line[1]}\n")

def main():
    args = define_and_parse_args()
    model_names, img_file, gt_pnt_file, out_dir = args
    # Load images and gt point cloud
    images = load_images(img_file)
    gt_pnts = load_point_cloud(gt_pnt_file)
    # Set up dir
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    shutil.copy(img_file, out_dir)
    shutil.copy(gt_pnt_file, out_dir)
    # List for saving 
    loss_values = []

    for model_name in model_names:
        # Load in model
        model: models.Sequential = models.load_model(
            model_name,
            custom_objects={'evaluate': loss.chamfer_distance.evaluate}
        )
        # Determine Loss and Predict point clouds
        test_loss = model.evaluate(images, gt_pnts)[0]
        predictions = model.predict(images)
        # Store results
        loss_values.append((os.path.split(model_name)[1], test_loss))
        save_predictions(predictions, f"{out_dir}/{os.path.split(model_name)[1]}_pnts.pickle")

    save_loss(loss_values, f"{out_dir}/loss_values.txt")

if __name__ == "__main__":
    main()