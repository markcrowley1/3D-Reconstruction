import cv2
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def define_and_parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-x", type=str, required=True)
    argparser.add_argument("-y", type=str, required=True)
    argparser.add_argument("-t", type=str, required=True)
    args = argparser.parse_args()
    parsed_args = [args.x, args.y, args.t]
    return parsed_args

def show_imgs(x: np.ndarray, y: np.ndarray):
    """Show training image pairs"""
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("Input")
    plt.imshow(x)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Expected Output")
    plt.imshow(y)

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
    t = pickle.load(file)
    file.close()

    print(len(x), len(y), len(t))
    for i in range(0, len(x), 120):
        print(t[i])
        show_imgs(x[i], y[i])
        
if __name__ == "__main__":
    main()