"""
Description:
    Prepare data to train view synthesis network.

    Generates and saves input tuple (input image and transform) along with
    target output image.
"""

import os
import cv2
import math
import json
import pickle
import random
import numpy as np
from vs_settings import *

def create_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    return

def save_data(data: np.ndarray, filename: str):
    pickle_out = open(filename, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()
    return

def save_batches(x: np.ndarray, t: np.ndarray, y: np.ndarray, batch_size: int, dir_name: str):
    # Make sure directories exist
    create_dir(dir_name)
    create_dir(f"{dir_name}/x/")
    create_dir(f"{dir_name}/t/")
    create_dir(f"{dir_name}/y/")
    # Save data in batch sized files
    num_batches = math.floor(len(x)/batch_size)
    for i in range(num_batches):
        low = i * batch_size
        high = low + batch_size
        x_batch = x[low:high]
        t_batch = t[low:high]
        y_batch = y[low:high]
        save_data(x_batch, f"{dir_name}/x/batch_{i}.pickle")
        save_data(t_batch, f"{dir_name}/t/batch_{i}.pickle")
        save_data(y_batch, f"{dir_name}/y/batch_{i}.pickle")
    return

def split_tuples(data_tuples: list[list]):
    # Lists for components of tuples
    x_imgs = []
    t_values = []
    y_imgs = []
    # Unpack items
    for x, t, y in data_tuples:
        x_imgs.append(x)
        t_values.append(t)
        y_imgs.append(y)
    # Reshape images
    x_imgs = np.array(x_imgs).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    y_imgs = np.array(y_imgs).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    return x_imgs, t_values, y_imgs

def read_metadata(dir: str):
    """Return each img file with elevation and azimuth"""
    metadata = []
    data = open(f"{dir}/rendering_metadata.txt", "r")
    filenames = open(f"{dir}/renderings.txt", "r")
    data = data.readlines()
    filenames = filenames.readlines()

    for i in range(len(filenames)):
        filename = filenames[i].strip()
        line = data[i].strip()
        azimuth = float(line.split()[0])
        elevation = float(line.split()[1])
        distance = float(line.split()[3])
        metadata.append((filename, elevation, azimuth, distance))

    return metadata

def calculate_change(a, b):
    """Calculate difference between angle a and angle b"""
    diff = b - a
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return diff

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_category_info(
        categories: list, 
        count: int, 
        taxonomy_file: str
    ) -> list[tuple]:
    """Gather info for selected categories and return as list"""
    output = []
    # Read taxonomy data
    file = open(taxonomy_file, "r")
    taxonomy = json.load(file)
    file.close()
    ids: dict = taxonomy["synsetIds"]
    num_instances: dict = taxonomy["numInstances"]
    # Get data for each selected category
    for category in categories:
        # Input error checking for each category
        if category not in ids.keys():
            print(f"{category} is not a base category of objects. Try one of:")
            print(list(ids.keys()))
            exit()
        if count > int(num_instances[category]):
            count = num_instances[category]
            print(f"Count exceeds numInstances, proceeding with {count} instances")

        # Set base dirs for object category
        render_base_dir = f"{RENDERINGS}/{ids[category]}"
        # Check for missing data dirs
        if not os.path.isdir(render_base_dir):
            exit(f"Exiting. Could not find {category} renderings directory.")
    	# Group category data
        output.append((ids[category], count, render_base_dir))
    return output

def main():
    # Get info for categories to generate data for
    category_info = parse_category_info(CATEGORIES, COUNT,
                                        "./nn_view_synthesis/category_info.json")
    create_dir(OUT_DIR)

    for i, cat_info in enumerate(category_info):
        # Unpack info for specific category
        id, count, render_base_dir = cat_info
        # Get names of subdirs containing renderings of models
        # Split paths to models into chunks - reduce concurrent RAM usage
        dir_names = os.listdir(render_base_dir)
        random.shuffle(dir_names)
        dir_names = dir_names[:count]
        dir_list = chunks(dir_names, 50)
        # Create paths to output dirs
        train_dir = f"{OUT_DIR}/train"
        val_dir = f"{OUT_DIR}/val"
        test_dir = f"{OUT_DIR}/test"
        # Create dirs if they don't exist
        create_dir(train_dir)
        create_dir(val_dir)
        create_dir(test_dir)
        # Create data for specific category
        for chunk, dir_names in enumerate(dir_list):
            paired_data = []
            for j, dir_name in enumerate(dir_names):
                render_dir = f"{render_base_dir}/{dir_name}/rendering"
                metadata = read_metadata(render_dir)
                for input_data in metadata:
                    # Unpack info for input image
                    input_img, input_elevation, input_azimuth, i_d = input_data
                    input_img_array = cv2.imread(f"{render_dir}/{input_img}")
                    input_img_array = cv2.cvtColor(input_img_array, cv2.COLOR_BGR2RGB)
                    input_img_array = cv2.resize(input_img_array, (IMG_SIZE,IMG_SIZE))
                    for output_data in metadata:
                        # Unpack info for output image
                        output_img, output_elevation, output_azimuth, o_d = output_data
                        if input_img != output_img:
                            # Read the target output image
                            output_img_array = cv2.imread(f"{render_dir}/{output_img}")
                            output_img_array = cv2.cvtColor(output_img_array,
                                                            cv2.COLOR_BGR2RGB)
                            output_img_array = cv2.resize(output_img_array,
                                                        (IMG_SIZE,IMG_SIZE))
                            # Calculate image transformation information
                            elevation = calculate_change(input_elevation, 
                                                        output_elevation)
                            azimuth = calculate_change(input_azimuth, output_azimuth)
                            transform = (elevation/10, azimuth/180, (o_d - i_d))
                            # Store data triplets
                            paired_data.append([input_img_array, transform,
                                                    output_img_array])
                # Progress message
                print(f"{i+1}th category --- {j + 1 + chunk*len(dir_names)}/{count}")

            # Seperate data into training, validation and test splits
            # Splitting data here ensures different models used for training and testing
            train_split_idx = math.floor(TRAINING_SPLIT * len(paired_data))
            val_split_idx = train_split_idx + math.floor(VALIDATION_SPLIT * len(paired_data))
            train_tuples = paired_data[:train_split_idx]
            val_tuples = paired_data[train_split_idx:val_split_idx]
            test_tuples = paired_data[val_split_idx:]
        
            # Shuffle training data so that data from same model is split up
            # Seperate training tuples into lists containing images and transforms
            # Save in batches
            random.shuffle(train_tuples)
            x_imgs, t_vals, y_imgs = split_tuples(train_tuples)
            cat_dir = f"{train_dir}/{id}_{chunk}"
            save_batches(x_imgs, t_vals, y_imgs, BATCH_SIZE, cat_dir)

            # Save validation data in batches as well
            x_imgs, t_vals, y_imgs = split_tuples(val_tuples)
            cat_dir = f"{val_dir}/{id}_{chunk}"
            save_batches(x_imgs, t_vals, y_imgs, BATCH_SIZE, cat_dir)

            # Save testing data in 3 pickle files (no need for batches)
            x_imgs, t_vals, y_imgs = split_tuples(test_tuples)
            cat_dir = f"{test_dir}/{id}_{chunk}"
            create_dir(cat_dir)
            save_data(x_imgs, f"{cat_dir}/x_imgs.pickle")
            save_data(t_vals, f"{cat_dir}/t_vals.pickle")
            save_data(y_imgs, f"{cat_dir}/y_imgs.pickle")

if __name__ == "__main__":
    main()