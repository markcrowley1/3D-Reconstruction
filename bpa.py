"""
Description:
    Initial implementation of the ball pivoting algorithm.

    Using the ball pivoting algorithm provided by open3d python module.
    Can be implemented with set of points contained within python list.
"""
import cv2
import open3d as o3d

def main():
    # Download bunny mesh, sample a set of points from it and visualise
    bunny = o3d.data.BunnyMesh()
    gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
    gt_mesh.compute_vertex_normals()

    pcd = gt_mesh.sample_points_poisson_disk(3000)
    o3d.visualization.draw_geometries([pcd])

    # Convert set of sampled points into 3d mesh and visualise
    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    o3d.visualization.draw_geometries([pcd, rec_mesh])

if __name__ == "__main__":
    main()