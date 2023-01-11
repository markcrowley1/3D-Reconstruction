"""
Description:
    Test script for manipulating Shapenet dataset
"""

import trimesh
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_graphics.datasets.shapenet import Shapenet
import tensorflow_graphics as tfg

def main():
    """Download Dataset"""
    shapenet = Shapenet()
    dataset = shapenet.load(
        split = "train",
        download_and_prepare_kwargs = {
            "download_config": tfds.download.DownloadConfig(manual_dir="D:\shapenet_base\shapenet_core")
        }
    )
    
    i = 0
    for example in dataset:
        i += 1
        trimesh, label, model_id = example['trimesh'], example['label'], example['model_id']
        if i > 5:
            break


if __name__ == "__main__":
    main()