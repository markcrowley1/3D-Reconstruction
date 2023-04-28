"""
Description:
    Constant values for training are set in this file, including:
        - Input/Output image size
        - Batch size
        - Epochs
        - Training Categories
        - Dataset size
"""
# Data location
RENDERINGS = "D:/ShapeNetRendering/ShapeNetRendering"
OUT_DIR = "examples/view_synthesis_data"
# Data specifications
CATEGORIES = ["car,auto,automobile,machine,motorcar"]
TRAINING_SPLIT = 0.7
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.2
COUNT = 10         # number of 3d models per category
# Input/Output Size
IMG_SIZE = 112
# Training parameters
EPOCHS = 5
BATCH_SIZE = 23