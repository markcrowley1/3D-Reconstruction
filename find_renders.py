import os
import json

# Location of shapenet dataset - must be downloaded manually
MANUAL_DIR = "D:\shapenet_base\shapenet_core"
# Set of object categories to be gathered for training
OBJ_LABELS = ["airplane,aeroplane,plane"]

def get_screenshot_folders(dir: str, num_models: int = 10) -> list:
    """Return paths of screenshot subdirs within a list"""
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
    print(ss_dirs[2])
    print(len(ss_dirs))

    for path in ss_dirs:
        base_path = os.path.split(path)[0]
        obj_file_path = get_obj_files(base_path)
        print(obj_file_path)

if __name__ == "__main__":
    main()