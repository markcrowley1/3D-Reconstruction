"""
Description:
    Example of image classification on cats and dogs.
    This script loads and preprocesses data

    Images downloaded from:
        https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
"""

import os
import cv2
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "D:\PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50

def show_image(img, path):
    # Read image in greyscale and resize
    img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
    img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))

    # Print img info and display img
    print(type(img_array))
    print(img_array.shape)
    plt.imshow(img_array, cmap="gray")
    plt.show()
    
def main():
    """Data loading & preprocessing"""
    # Create training data from set of images
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:   
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([img_array, class_num])
            except Exception as e:
                pass
    print(len(training_data))

    # Shuffle the training data
    # Seperate the imgs and labels into lists
    X = []
    Y = []
    random.shuffle(training_data)
    for features, label in training_data:
        X.append(features)
        Y.append(label)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Save data after processing
    pickle_out = open("examples/x.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("examples/y.pickle", "wb")
    pickle.dump(Y, pickle_out)
    pickle_out.close()
    
if __name__ == "__main__":
    main()