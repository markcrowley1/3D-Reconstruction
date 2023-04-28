"""
Description:
    Constant values for training are set in this file, including:
        - Input image size
        - Point Set size
        - Batch size
        - Epochs
        - Training Categories
        - Dataset size
"""
# Data location
SHAPENETCORE = "D:/shapenet_base/shapenet_obj"
RENDERINGS = "D:/ShapeNetRendering/ShapeNetRendering"
OUT_DIR = "D:/shapenet_base/table_data"
# Data specifications
CATEGORIES = ["table"]
TRAINING_SPLIT = 0.8
COUNT = 500         # number of 3d models per category
# Input/Output Size
IMG_SIZE = 112
POINTS = 1024
# Training parameters
EPOCHS = 20
BATCH_SIZE = 24