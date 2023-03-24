"""
Description:
    Script used to move obj files from shapenet dir to seperate dir.
    This is done due to bug in open3d mesh loading function, textures cause
    errors that force the program to shut down regardless of try/exception.
"""

import os
import shutil

SHAPENETCORE = "D:/shapenet_base/shapenet_core"
RENDERINGS = "D:/ShapeNetRendering/ShapeNetRendering"

def main():
    new_base = "D:/shapenet_base/shapenet_obj"
    os.mkdir(new_base)
    for category in os.listdir(RENDERINGS):
        os.mkdir(f"{new_base}/{category}")
        for model in os.listdir(f"{SHAPENETCORE}/{category}"):
            try:
                obj_file = f"{SHAPENETCORE}/{category}/{model}/models/model_normalized.obj"
                new_dir = f"{new_base}/{category}/{model}"
                new_models_dir = f"{new_base}/{category}/{model}/models"
                os.mkdir(new_dir)
                os.mkdir(new_models_dir)
                shutil.copy(obj_file, new_models_dir)
            except:
                pass

if __name__ == "__main__":
    main()