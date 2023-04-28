"""
Description:
    Convert point clouds to meshes and visualise.
"""

import pickle
import meshzoo
import trimesh
import argparse
import numpy as np
from scipy import spatial

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

def main():
    arg = define_and_parse_args()
    point_sets = load_point_cloud(arg)

    for point_set in point_sets:    
        verts, cells = meshzoo.icosa_sphere(8)
        print(verts.shape)
        
        tree = spatial.KDTree(verts)
        distances, indices = tree.query(point_set)
        for i, index in enumerate(indices):
            verts[i] = point_set[index]

        # Create trimesh mesh
        mesh = trimesh.Trimesh(verts, cells)
        mesh.show()
    # # Save to obj file
    # trimesh.exchange.export.export_mesh(mesh, "test.obj")
        
if __name__ == "__main__":
    main()