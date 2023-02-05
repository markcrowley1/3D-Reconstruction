
import numpy as np
import trimesh
import math
import matplotlib.pyplot as plt

def visualise_points(points):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:,0], points[:,1], points[:,2], s=10)
    ax.set_axis_off()
    plt.show()


obj_file = "D:/shapenet_base/shapenet_core/02691156/1a6ad7a24bb89733f412783097373bdc/models/model_normalized.obj"
scene = trimesh.load(obj_file, force="scene")
corners = scene.bounds
scene.show()
r_e = trimesh.transformations.euler_matrix(
    math.radians(90),
    math.radians(0),
    math.radians(0),
    "ryxz",
)

t_r = scene.camera.look_at(corners, rotation=r_e)
scene.camera_transform = t_r
scene.show()
# png = scene.save_image()
# with open("test.png", "wb") as file:
#     file.write(png)
#     file.close()