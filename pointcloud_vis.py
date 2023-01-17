
import sys
import pickle
import matplotlib.pyplot as plt

def visualise_points(points):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:,0], points[:,1], points[:,2], s=100)
    ax.set_axis_off()
    plt.show()

def main():
    # Get name of pickle file containing point cloud
    filename = sys.argv[1]
    # Load and display saved point cloud as test
    with open(filename, "rb") as file:
        points = pickle.load(file)
    file.close()
    visualise_points(points)

if __name__ == "__main__":
    main()