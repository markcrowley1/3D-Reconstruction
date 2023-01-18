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
import json
import trimesh
import pickle
import numpy as np

OBJ_LABELS = ["airplane,aeroplane,plane"]
MANUAL_DIR = "D:\shapenet_base\shapenet_core"

SAMPLES = 1024*10

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
    numInstances = 0
    for item in data:
        if item["name"] in OBJ_LABELS:
            id = item["synsetId"]
            numInstances = item["numInstances"]
            category_dir = os.path.join(MANUAL_DIR, id)
            break

    ss_dirs = get_screenshot_folders(category_dir)
    print(len(ss_dirs))
    
    data_dir = "./data"
    if os.path.exists(data_dir) == False:
        os.mkdir(data_dir)

    i = 0
    for ss_dir in ss_dirs:
        base_path = os.path.split(ss_dir)[0]
        model_path = get_obj_files(base_path)[0]

        # Load in .obj as mesh and sample a set of points
        mesh = trimesh.load(model_path, force="mesh")
        sample = trimesh.sample.sample_surface(mesh, SAMPLES)
        sampled_points = sample[0]
        points = np.array(sampled_points)

        # Save the sampled set of points in pickle files
        i += 1
        data_subdir = f"{data_dir}/subdir{i}"
        if os.path.exists(data_subdir) == False:
            os.mkdir(data_subdir)
        with open(f"{data_subdir}/points.pickle", "wb") as file:
            pickle.dump(points, file)
        file.close()

        # Save obj file for visual comparison
        shutil.copy(model_path, data_subdir)

        # Save the screenshot images in subfolder
        image_folder = f"{data_subdir}/screenshots"
        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)
        shutil.copytree(ss_dir, image_folder)

if __name__ == "__main__":
    main()