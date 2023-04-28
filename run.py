"""
Description:
    Run inference on an input image. Top level script for framework.
"""

import os
import argparse
import numpy as np
from source import MESH, PSG, TEXGEN, VIS

def define_and_parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-img", type=str, required=True)
    argparser.add_argument("-o", type=str, default="./mesh")
    args = argparser.parse_args()
    return args.img, args.o

def main():
    # Get arguments
    img, outdir = define_and_parse_args()
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    # Load classes that handle nn, mesh, texture and visualisation
    psg = PSG("models/5cat_20e_2")
    mesh_handler = MESH()
    tex_handler = TEXGEN()
    visualiser = VIS()
    # Process image and save results
    point_set: np.ndarray = psg.generate_pointset(img)
    mesh = mesh_handler.create_mesh(point_set)
    mesh = mesh_handler.process_mesh(mesh, 1, 1, 0)
    mesh_handler.save_mesh(mesh, "texture.png", outdir)

    img_array = tex_handler.read_img(img)
    texture = tex_handler.generate_texture(img_array)
    tex_handler.save_img(img_array[:,:,::-1], f"{outdir}/original_image.png")
    tex_handler.save_img(texture, f"{outdir}/texture.png")

    # Visualisation
    visualiser.visualise_prediction(img_array, point_set)

if __name__ == "__main__":
    main()