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
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from settings import *

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

def parse_category_info(
        categories: list, 
        count: int, 
        taxonomy_file: str
    ) -> list[tuple]:
    """Gather info for selected categories and return as list"""
    output = []
    # Read taxonomy data
    file = open(taxonomy_file, "r")
    taxonomy = json.load(file)
    file.close()
    ids: dict = taxonomy["synsetIds"]
    num_instances: dict = taxonomy["numInstances"]
    # Get data for each selected category
    for category in categories:
        # Input error checking for each category
        if category not in ids.keys():
            print(f"{category} is not a base category of objects. Try one of:")
            print(list(ids.keys()))
            exit()
        if count > int(num_instances[category]):
            count = num_instances[category]
            print(f"Count exceeds numInstances, proceeding with {count} instances")

        # Set base dirs for object category
        render_base_dir = f"{RENDERINGS}/{ids[category]}"
        model_base_dir = f"{SHAPENETCORE}/{ids[category]}"
        # Check for missing data dirs
        if not os.path.isdir(render_base_dir):
            exit(f"Exiting. Could not find {category} renderings directory.")
        if not os.path.isdir(model_base_dir):
            exit(f"Exiting. Could not find {category} models directory.")
    	# Group category data
        output.append((count, render_base_dir, model_base_dir))
    return output

def save_data(data, training_split: float, dir: str):
    """Seperate data into training and test splits and save."""
    # Shuffle pairs of training data
    imgs = []
    point_sets = []
    random.shuffle(data)

    # Seperate images and point clouds for storage in memory
    for img, point_set in data:
        imgs.append(img)
        point_sets.append(point_set)
    imgs = np.array(imgs).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    # Split images and point sets into training and testing splits
    split_index = math.floor(training_split * len(imgs))
    training_images, test_images = imgs[:split_index], imgs[split_index:]
    training_ps, test_ps = point_sets[:split_index], point_sets[split_index:]

    # Save data
    if not os.path.isdir(dir):
        os.mkdir(dir)

    pickle_out = open(f"{dir}/train_imgs.pickle", "wb")
    pickle.dump(training_images, pickle_out)
    pickle_out.close()

    pickle_out = open(f"{dir}/test_imgs.pickle", "wb")
    pickle.dump(test_images, pickle_out)
    pickle_out.close()

    pickle_out = open(f"{dir}/train_pnts.pickle", "wb")
    pickle.dump(training_ps, pickle_out)
    pickle_out.close()

    pickle_out = open(f"{dir}/test_pnts.pickle", "wb")
    pickle.dump(test_ps, pickle_out)
    pickle_out.close()

def main():
    # Get info for categories to generate data for
    category_info = parse_category_info(CATEGORIES, COUNT,
                                        "./training/category_info.json")
    paired_data = []
    for i, cat_info in enumerate(category_info):
        # Unpack info for specific category
        count, render_base_dir, model_base_dir = cat_info
        # Get names of subdirs containing models and randomly select COUNT models
        dir_names = os.listdir(render_base_dir)
        random.shuffle(dir_names)
        dir_names = dir_names[:count]
        
        for j, dir_name in enumerate(dir_names):
            # Handle paths
            render_dir = f"{render_base_dir}/{dir_name}/rendering"
            model_filename = f"{model_base_dir}/{dir_name}/models/model_normalized.obj"

            # Sample the 3d mesh and orient to base position
            try:
                mesh = o3d.io.read_triangle_mesh(model_filename)
                pcd = mesh.sample_points_poisson_disk(number_of_points=POINTS, init_factor=5)
                points = np.asarray(pcd.points)
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
                # Progress message
                print(f"{i+1}th category --- {j+1}/{count}")
            except:
                pass

    # Split train and test data and save in pickle same dir
    save_data(paired_data, TRAINING_SPLIT, OUT_DIR)

if __name__ == "__main__":
    main()