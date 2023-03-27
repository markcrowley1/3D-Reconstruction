
import numpy as np
import matplotlib.pyplot as plt

class VIS:
    def __init__(self):
        return
    
    def visualise_prediction(self, img: np.ndarray, points: np.ndarray):
        # Show rendering of textured 3D mesh
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 2, 1)
        ax.set_title("Rendering")
        plt.imshow(img)

        # Show corresponding ground truth 3d point cloud
        ax = fig.add_subplot(1, 2, 2, projection="3d")
        ax.scatter(points[:,0], points[:,1], points[:,2], s=5, c="red")
        ax.set_axis_on()
        ax.set_aspect("equal")
        ax.set_title("Point Cloud")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_zticklabels([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.view_init(0, 0)

        plt.show()