"""
Description:
    Manipulating shapenet dataset using non tensorflow packages.
"""

import os
import json
import pickle
import trimesh
import cv2
from matplotlib import pyplot as plt

# Location of shapenet dataset - must be downloaded manually
MANUAL_DIR = "D:\shapenet_base\shapenet_core"
# Set of object categories to be gathered for training
OBJ_LABELS = ["bag,traveling bag,travelling bag,grip,suitcase"]

def determine_base_categories(data: list):
    """Determine names of base categories"""
    for name in os.listdir(MANUAL_DIR):
        path = os.path.join(MANUAL_DIR, name)
        if os.path.isdir(path):
            id = os.path.split(path)[1]
            for item in data:
                if item["synsetId"] == id:
                    print(item["name"] + " " + str(item["numInstances"]))
                    break

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

def visualise_points(points):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:,0], points[:,1], points[:,2], s=100)
    ax.set_axis_off()
    plt.show()

def main():
    """Create set of training data based on taxonomy.json file"""
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
    print(numInstances)

    # Create dir for data
    data_dir = "./data"
    if os.path.exists(data_dir) == False:
        os.mkdir(data_dir)

    # Create list of .obj paths
    # Loop through .obj files and generate training data
    model_paths = get_obj_files(category_dir)
    i = 0
    for model_path in model_paths:
        # Load in .obj as mesh and sample a set of points
        mesh = trimesh.load(model_path, force="mesh")
        sample = trimesh.sample.sample_surface(mesh, 1024)
        points = sample[0]
        # scene.show()
        # visualise_points(points)

        # Save the sampled set of points in pickle files
        i += 1
        data_subdir = f"{data_dir}/subdir{i}"
        if os.path.exists(data_subdir) == False:
            os.mkdir(data_subdir)
        with open(f"{data_subdir}/points.pickle", "wb") as file:
            pickle.dump(points, file)
        file.close()

        # Generate a set of RGB images for each .obj file
        # Network can train on images as we have ground truth point cloud

    # Load and display saved point cloud as test
    with open(f"{data_subdir}/points.pickle", "rb") as file:
        points = pickle.load(file)
    file.close()
    visualise_points(points)

if __name__ == "__main__":
    main()