"""
Description:
    Class handles mesh generation from point set input. Steps include:
        - Generating Mesh.
        - Mapping Texture to Mesh.
        - Saving Textured Mesh with obj and mtl files.
"""

import os
import trimesh
import numpy as np
import open3d as o3d

class MESH:
    def __init__(self):
        pass

    def create_mesh(self, point_set: np.ndarray):
        """Input point set and returns mesh"""
        # Load point cloud and estimate normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_set)
        # Estimate mesh from point cloud
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, 0.06)
        return mesh
    
    def process_mesh(
        self,
        mesh,
        nlaplacian: int = 0,
        naverage: int = 0,
        ntaubin: int = 0,
    )-> trimesh.Trimesh:
        """Process mesh (smoothing and conversion to trimesh obj) and return"""
        # Apply smoothing filters according to args
        if nlaplacian > 0:
            mesh = mesh.filter_smooth_laplacian(number_of_iterations=nlaplacian)
            mesh.compute_vertex_normals()
        if naverage > 0:
            mesh = mesh.filter_smooth_simple(number_of_iterations=naverage)
            mesh.compute_vertex_normals()
        if ntaubin > 0:
            mesh = mesh.filter_smooth_taubin(number_of_iterations=ntaubin)
            mesh.compute_vertex_normals()
        # Convert to trimesh representation
        mesh = trimesh.Trimesh(np.asarray(mesh.vertices),
                               np.asarray(mesh.triangles),
                               vertex_normals=np.asarray(mesh.vertex_normals))
        mesh.fix_normals()
        return mesh

    def save_mesh(self, mesh: trimesh.Trimesh, texture_file: str, outdir: str):
        """Save textured mesh with obj and mtl file pair."""
        filename = f"{outdir}/model.obj"
        # Write the obj file
        with open(filename, 'w') as f:
            # Write the mtl reference to the file
            f.write('mtllib model.mtl\n')

            # Write the vertices to the file
            for vertex in mesh.vertices:
                f.write('v {} {} {}\n'.format(vertex[0], vertex[1], vertex[2]))

            # Write the vertex normals to the file
            for normal in mesh.vertex_normals:
                f.write('vn {} {} {}\n'.format(normal[0], normal[1], normal[2]))

            # Write the material definition to the file
            f.write('usemtl material\n')

            # Write the faces to the file
            faces = mesh.faces + 1
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
            f.write('map_Kd {}\n'.format(texture_file))