"""
Description:
    Test script for manipulating shapenet dataset with tensorflow tools
"""

import trimesh
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_graphics.datasets.shapenet import Shapenet
import tensorflow_graphics as tfg

# Location of shapenet dataset - must be downloaded manually
MANUAL_DIR = "D:\shapenet_base\shapenet_core"

def main():
    """Attempt to prepare data using builder config tf"""
    tfg.datasets.shapenet.Shapenet.builder_config_cls(
        model_subpath='models/model_normalized.obj'
    )

    """Organises dataset but stores all data again on C drive"""
    # dataset = shapenet.load(
    #     split = "train",
    #     download_and_prepare_kwargs = {
    #         "download_config": tfds.download.DownloadConfig(manual_dir=MANUAL_DIR)
    #     }
    # )
    
    # i = 0
    # for example in dataset:
    #     i += 1
    #     trimesh, label, model_id = example['trimesh'], example['label'], example['model_id']
    #     print(label)
    #     print(model_id)
    #     if i > 5:
    #         break


if __name__ == "__main__":
    main()