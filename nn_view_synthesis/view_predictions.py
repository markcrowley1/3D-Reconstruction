"""
Description:
    View the input image, target image and predicted image side by side
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def define_and_parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-x", type=str, required=True)
    argparser.add_argument("-y", type=str, required=True)
    argparser.add_argument("-p", type=str, required=True)
    args = argparser.parse_args()
    parsed_args = [args.x, args.y, args.t]
    return parsed_args

def show_imgs(x: np.ndarray, y: np.ndarray, p: np.ndarray):
    """Show training image pairs"""
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title("Input")
    plt.imshow(x)

    ax = fig.add_subplot(1, 3, 2)
    ax.set_title("Target Output")
    plt.imshow(y)

    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("Network Output")
    plt.imshow(p)

    plt.show()

def main():
    # Read in images
    paths = define_and_parse_args()

    file = open(paths[0], "rb")
    x = np.array(pickle.load(file))
    x = x/255.0
    file.close()

    file = open(paths[1], "rb")
    y = np.array(pickle.load(file))
    y = y/255.0
    file.close()

    file = open(paths[2], "rb")
    p = np.array(pickle.load(file))
    p = p/255.0
    file.close()

    # Skip through and show results at an interval
    for i in range(0, len(x), 120):
        show_imgs(x[i], y[i], p[i])
        
if __name__ == "__main__":
    main()