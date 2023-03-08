"""
Description:
    Convert point clouds to meshes and visualise.
"""

import os
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

def write_obj_file_with_texture_and_mtl(filename, vertices, faces, normals, texture_coords, texture_filename):
    """
    Writes an obj file with texture mapping and a corresponding mtl file given the vertices, faces, normals, texture coordinates,
    and texture filename as inputs.

    :param filename: The name of the obj file to be written.
    :param vertices: A list of vertices in the format [(x1, y1, z1), (x2, y2, z2), ...].
    :param faces: A list of faces, where each face is a list of vertex indices in the format [(v1, v2, v3), (v4, v5, v6), ...].
    :param normals: A list of vertex normals in the format [(nx1, ny1, nz1), (nx2, ny2, nz2), ...].
    :param texture_coords: A list of texture coordinates in the format [(u1, v1), (u2, v2), ...].
    :param texture_filename: The name of the jpg image to be used as the texture.
    """
    # Write the obj file
    with open(filename, 'w') as f:
        # Write the mtl reference to the file
        f.write('mtllib {}.mtl\n'.format(os.path.splitext(filename)[0]))

        # Write the vertices to the file
        for vertex in vertices:
            f.write('v {} {} {}\n'.format(vertex[0], vertex[1], vertex[2]))

        # Write the texture coordinates to the file
        for texture_coord in texture_coords:
            f.write('vt {} {}\n'.format(texture_coord[0], texture_coord[1]))

        # Write the vertex normals to the file
        for normal in normals:
            f.write('vn {} {} {}\n'.format(normal[0], normal[1], normal[2]))

        # Write the material definition to the file
        f.write('usemtl material\n')

        # Write the faces to the file
        faces = faces + 1
        for face in faces:
            # Each face is a list of vertex indices, texture indices, and normal indices in the format [(v1, t1, n1), (v2, t2, n2), (v3, t3, n3)]
            # We need to subtract 1 from the indices to convert them to 0-indexed
            f.write(f'f {face[0]}//{face[0]} {face[1]}//{face[1]} {face[2]}//{face[2]}\n')

    # Write the mtl file
    with open('{}.mtl'.format(os.path.splitext(filename)[0]), 'w') as f:
        # Write the material name to the file
        f.write('newmtl material\n')

        # Additional material settings
        f.write('Kd 1 1 1\n')
        f.write('Ka 0 0 0\n')
        f.write('Ks 0.4 0.4 0.4\n')
        f.write('Ke 0 0 0\n')
        f.write('Ns 10\n')
        f.write('illum 2\n')

        # Write the texture filename to the file
        f.write('map_Kd {}\n'.format(texture_filename))

def main():
    arg = define_and_parse_args()
    point_sets = load_point_cloud(arg)

    for point_set in point_sets:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_set)
        pcd.estimate_normals()

        # distances = pcd.compute_nearest_neighbor_distance()
        # avg_dist = np.mean(distances)
        # print(avg_dist)
        # radius = 1.5 * avg_dist   
        # radii = [0.005, 0.01, 0.015, 0.02, 0.1]
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.07)
        mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), vertex_normals=np.asarray(mesh.vertex_normals))
        visualise_points(point_set)
        # mesh.fix_normals()
        trimesh.smoothing.filter_humphrey(mesh=mesh)
        mesh.show()
        write_obj_file_with_texture_and_mtl("test.obj", mesh.vertices, mesh.faces, mesh.vertex_normals, [], "texture2.jpg")
        trimesh.exchange.export.export_mesh(mesh, "trimesh_save.obj")
        file_contents: str = trimesh.exchange.obj.export_obj(mesh, mtl_name="test.mtl")
        file = open("trimesh_save2.obj", "w")
        file.write(file_contents)
        file.close()
        break
        
if __name__ == "__main__":
    main()