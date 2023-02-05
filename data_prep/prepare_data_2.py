"""
Description:
    Script for grouping together training data for specific object categories.

    ShapenetCore Dataset at:
        https://shapenet.org/download/shapenetcore

    Image Renderings at:
        http://ftp.cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz
"""

import os
import shutil

import cv2
import json
import pickle
import random
import trimesh
import numpy as np
import matplotlib.pyplot as plt

OBJ_LABELS = ["airplane,aeroplane,plane"]
MANUAL_DIR = "D:\shapenet_base\shapenet_core"

IMG_SIZE = 64
SAMPLES = 256
NUM_MODELS = 100

def get_screenshot_folders(dir: str, num_models: int = 10) -> list:
    """Recursive function to return paths of screenshot subdirs."""
    ss_dir = []
    for name in os.listdir(dir):
        if len(ss_dir) >= num_models:
            break
        path = os.path.join(dir, name)
        if os.path.isdir(path) and os.path.split(path)[1] == "screenshots":
            ss_dir.append(path)
        elif os.path.isdir(path):
            paths = get_screenshot_folders(path)
            ss_dir = ss_dir + paths

    return ss_dir

def get_obj_files(dir: str) -> list:
    """Recursive function that finds and returnes paths to .obj files"""
    model_paths = []
    for name in os.listdir(dir):
        path = os.path.join(dir, name)
        if os.path.isdir(path):
            paths = get_obj_files(path)
            model_paths = model_paths + paths
        elif os.path.split(path)[1] == "model_normalized.obj":
            model_paths.append(path)

    return model_paths

def main():
    """Find the folders with screenshots for training"""
    # Read taxonomy.json
    taxonomy = os.path.join(MANUAL_DIR, "taxonomy.json")
    with open(taxonomy, "r") as file:
        data: list = json.load(file)
    file.close()

    # Find directory corresponding to specified obj category
    for item in data:
        if item["name"] in OBJ_LABELS:
            id = item["synsetId"]
            category_dir = os.path.join(MANUAL_DIR, id)
            break
    ss_dirs = get_screenshot_folders(category_dir, NUM_MODELS)
    print(len(ss_dirs))
    
    # Create directories to hold data
    data_dir = "./data"
    if os.path.exists(data_dir) == False:
        os.mkdir(data_dir)

    mesh_folder = f"{data_dir}/meshes"
    if os.path.exists(mesh_folder) == False:
        os.mkdir(mesh_folder)

    """Gather Training data"""
    training_data = []
    mesh_folder = f"{data_dir}/meshes"
    for i, ss_dir in enumerate(ss_dirs):
        # Get object files
        base_path = os.path.split(ss_dir)[0]
        model_path = get_obj_files(base_path)[0]

        # Load in .obj as mesh and sample a set of points
        mesh = trimesh.load(model_path, force="mesh")
        sample = trimesh.sample.sample_surface(mesh, SAMPLES)
        points = np.array(sample[0])

        # Save obj files for visual comparison
        data_subdir = f"{mesh_folder}/subdir{i}"
        if os.path.exists(data_subdir) == False:
            os.mkdir(data_subdir)
        shutil.copy(model_path, data_subdir)
        
        # Create training data
        for filename in os.listdir(ss_dir):
            if filename.endswith(".png"):
                # Read image in greyscale and resize
                img_array = cv2.imread(os.path.join(ss_dir, filename), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([img_array, points])

    # Shuffle training data
    x = []
    y = []
    random.shuffle(training_data)

    # Seperate training data and ground truth testing data
    for img, pc in training_data:
        x.append(img)
        y.append(pc)
    x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Save data after processing
    pickle_out = open("./data/images.pickle", "wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()

    pickle_out = open("./data/points.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

if __name__ == "__main__":
    main()