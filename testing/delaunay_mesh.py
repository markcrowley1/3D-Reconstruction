"""
Description:
    Convert point clouds to meshes and visualise.
"""

import pickle
import trimesh
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def define_and_parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-f", type=str, default="./output/nn_output.pickle")
    args = argparser.parse_args()
    return args.f

def load_point_cloud(filename: str) -> np.ndarray:
    # Load in point clouds
    file = open(filename, "rb")
    point_sets = np.array(pickle.load(file))
    file.close()
    return point_sets

def visualise_points(points):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:,0], points[:,1], points[:,2], s=10)
    ax.set_axis_off()
    plt.show()

def main():
    arg = define_and_parse_args()
    point_sets = load_point_cloud(arg)

    for point_set in point_sets:
        break
    pnts = point_sets[1]
    tri = Delaunay(point_set)
    simplex = tri.simplices
    faces = [simplex[0]+1, simplex[1]+1, simplex[2]+1]
    mesh = trimesh.Trimesh(pnts, faces)
    mesh.show()
        
if __name__ == "__main__":
    main()