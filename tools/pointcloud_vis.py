
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

def visualise_points(points):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:,0], points[:,1], points[:,2], s=10)
    # ax.set_axis_off()
    plt.show()

def main():
    # Get name of pickle file containing point cloud
    # Load and display saved point cloud as test
    filename = sys.argv[1]
    file = open(filename, "rb")
    points = np.array(pickle.load(file))
    file.close()

    print(type(points))
    print(points.shape)
    for pc in points:
        print(type(pc))
        print(pc.shape)
        visualise_points(pc)

if __name__ == "__main__":
    main()