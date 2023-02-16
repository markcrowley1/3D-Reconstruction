"""
Description:
    Convert point clouds to meshes and visualise.
"""

import pickle
import trimesh
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

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
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_set)
        pcd.estimate_normals()

        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        print(avg_dist)
        radius = 1.5 * avg_dist   
        radii = [0.005, 0.01, 0.015, 0.02, 0.1]

        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.07)
        mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), vertex_normals=np.asarray(mesh.vertex_normals))

        print(point_set.shape)
        visualise_points(point_set)
        mesh.fix_normals()
        trimesh.smoothing.filter_humphrey(mesh=mesh)
        mesh.show()
        
if __name__ == "__main__":
    main()