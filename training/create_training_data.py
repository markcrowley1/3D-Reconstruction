"""
Description:
    Generate ground truth point clouds for the images in shapenet renderings
    dataset and provide a one to one mapping
"""

import os
import cv2
import math
import json
import pickle
import random
import trimesh
import argparse
import numpy as np
from scipy.spatial.transform import Rotation

SHAPENETCORE = "D:/shapenet_base/shapenet_core"
RENDERINGS = "D:/ShapeNetRendering/ShapeNetRendering"

IMG_SIZE = 112
POINTS = 1024

def define_and_parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", type=str, default="airplane,aeroplane,plane")
    argparser.add_argument("-ts", type=float, default=0.8)
    argparser.add_argument("-n", type=int, default=10)
    args = argparser.parse_args()
    if args.ts > 1 or args.ts <= 0:
        exit("Training split arg not valid - must be between 0 and 1")
    if args.n <= 0:
        exit("Number of training instances must be positive integer")
    parsed_args = {"category": args.c, "train_split": args.ts, "num": args.n}
    return parsed_args

def read_metadata(dir: str) -> list[tuple]:
    """Return each img file with elevation and azimuth"""
    metadata = []
    data = open(f"{dir}/rendering_metadata.txt", "r")
    filenames = open(f"{dir}/renderings.txt", "r")
    data = data.readlines()
    filenames = filenames.readlines()

    for i in range(len(filenames)):
        filename = filenames[i].strip()
        line = data[i].strip()
        azimuth = float(line.split()[0])
        elevation = float(line.split()[1])
        metadata.append((filename, elevation, azimuth))

    return metadata

def rotate(points: np.ndarray, rotation: list) -> np.ndarray:
    """Rotate point set using euler matrix"""
    r = Rotation.from_euler("xyz", rotation, degrees= True)
    r = np.array(r.as_matrix())
    points = np.matmul(points, r)
    return points

def main():
    # Read args - object category, train/test split and number of instances
    args = define_and_parse_args()
    category, training_split, count = (args["category"], args["train_split"],
                                       args["num"])
    # Read taxonomy data
    file = open("category_info.json", "r")
    taxonomy = json.load(file)
    file.close()
    num_instances: dict = taxonomy["numInstances"]
    ids: dict = taxonomy["synsetIds"]
    
    # Input error checking
    if category not in ids.keys():
        print(f"{category} is not a base category of objects. Try one of:")
        print(list(ids.keys()))
        exit()

    if count > int(num_instances[category]):
        count = num_instances[category]
        print(f"Count exceeds numInstances, proceeding with {count} instances")

    # Check for missing data dirs
    render_base_dir = f"{RENDERINGS}/{ids[category]}"
    model_base_dir = f"{SHAPENETCORE}/{ids[category]}"

    if not os.path.isdir(render_base_dir):
        exit(f"Exiting. Could not find {category} renderings directory.")

    if not os.path.isdir(model_base_dir):
        exit(f"Exiting. Could not find {category} models directory.")

    # Loop through each rendering dir and match with model dir
    paired_data = []
    dir_names = os.listdir(render_base_dir)
    random.shuffle(dir_names)
    dir_names = dir_names[:count]
    
    for dir_name in dir_names:
        # Handle paths
        render_dir = f"{render_base_dir}/{dir_name}/rendering"
        model_filename = f"{model_base_dir}/{dir_name}/models/model_normalized.obj"

        # Sample the 3d mesh and orient to base position
        mesh = trimesh.load(model_filename, force="mesh")
        sample = trimesh.sample.sample_surface(mesh, POINTS)
        sampled_points = sample[0]
        points = np.array(sampled_points)
        base_points = rotate(points, [-90, 90, 0])

        # Get orientation information and apply to corresponding point clouds
        metadata = read_metadata(render_dir)
        for data in metadata:
            img_filename = f"{render_dir}/{data[0]}"
            img_array = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))

            rotation = [0, -data[1], -data[2]]
            points = rotate(base_points, rotation)
            paired_data.append([img_array, points])

    # Shuffle pairs of training data
    imgs = []
    point_sets = []
    random.shuffle(paired_data)

    # Seperate images and point clouds for storage in memory
    for img, point_set in paired_data:
        imgs.append(img)
        point_sets.append(point_set)
    imgs = np.array(imgs).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    # Split images and point sets into training and testing splits
    split_index = math.floor(training_split * len(imgs))
    training_images, test_images = imgs[:split_index], imgs[split_index:]
    training_ps, test_ps = point_sets[:split_index], point_sets[split_index:]

    # Save data
    if not os.path.isdir("./data"):
        os.mkdir("data")

    pickle_out = open("./data/train_imgs.pickle", "wb")
    pickle.dump(training_images, pickle_out)
    pickle_out.close()

    pickle_out = open("./data/test_imgs.pickle", "wb")
    pickle.dump(test_images, pickle_out)
    pickle_out.close()

    pickle_out = open("./data/train_pnts.pickle", "wb")
    pickle.dump(training_ps, pickle_out)
    pickle_out.close()

    pickle_out = open("./data/test_pnts.pickle", "wb")
    pickle.dump(test_ps, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    main()