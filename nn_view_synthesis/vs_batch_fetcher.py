"""
Description:
    Class for loading in batches of data for training/validation.
    Batch size is the number of examples in each pickle file (24).
"""

import os
import keras
import pickle
import numpy as np

def gen_file_list(dir_name: str):
    # Dir names
    train_dir = f"{dir_name}/train"
    val_dir = f"{dir_name}/val"
    test_dir = f"{dir_name}/test"
    # Loop through training dir
    x_imgs = []
    t_vals = []
    y_imgs = []
    for category in os.listdir(train_dir):
        cat_dir = f"{train_dir}/{category}"
        for directory in os.listdir(cat_dir):
            if directory == "x":
                x_imgs = x_imgs + [f"{cat_dir}/{directory}/{name}" for name in os.listdir(f"{cat_dir}/{directory}")]
            elif directory == "t":
                t_vals = t_vals + [f"{cat_dir}/{directory}/{name}" for name in os.listdir(f"{cat_dir}/{directory}")]
            elif directory == "y":
                y_imgs = y_imgs + [f"{cat_dir}/{directory}/{name}" for name in os.listdir(f"{cat_dir}/{directory}")]

    train_files = list(zip(x_imgs, t_vals, y_imgs))

    # Loop through validation dir
    x_imgs = []
    t_vals = []
    y_imgs = []
    for category in os.listdir(val_dir):
        cat_dir = f"{val_dir}/{category}"
        for directory in os.listdir(cat_dir):
            if directory == "x":
                x_imgs = x_imgs + [f"{cat_dir}/{directory}/{name}" for name in os.listdir(f"{cat_dir}/{directory}")]
            elif directory == "t":
                t_vals = t_vals + [f"{cat_dir}/{directory}/{name}" for name in os.listdir(f"{cat_dir}/{directory}")]
            elif directory == "y":
                y_imgs = y_imgs + [f"{cat_dir}/{directory}/{name}" for name in os.listdir(f"{cat_dir}/{directory}")]

    val_files = list(zip(x_imgs, t_vals, y_imgs))

    # Loop through testing dir
    x_imgs = []
    t_vals = []
    y_imgs = []
    for category in os.listdir(test_dir):
        cat_dir = f"{test_dir}/{category}"
        for file in os.listdir(cat_dir):
            if file == "x_imgs.pickle":
                x_imgs.append(f"{cat_dir}/{file}")
            elif file == "t_vals.pickle":
                t_vals.append(f"{cat_dir}/{file}")
            elif file == "y_imgs.pickle":
                y_imgs.append(f"{cat_dir}/{file}")

    test_files = list(zip(x_imgs, t_vals, y_imgs))

    return train_files, val_files, test_files

class BatchFetcher(keras.utils.Sequence):

    def __init__(self, training_data, batch_size, shuffle=True):
        """Initialisation"""
        self.data_tuples = training_data
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):
        """Fetch one batch of data"""
        # Get index and filenames for current batch
        index = self.indexes[idx]
        x_file, t_file, y_file = self.data_tuples[index]

        # Load images and corresponding point sets
        x = self.__load_data(x_file)
        t = self.__load_data(t_file)
        y = self.__load_data(y_file)

        # Normalise images
        x = x/255.0
        y = y/255.0

        return [x, t], y

    def on_epoch_end(self):
        """Shuffle order of data at end of each epoch"""
        self.indexes = np.arange(len(self.data_tuples))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __load_data(self, filename: str):
        """Load imgs or point sets from pickle file"""
        file = open(filename, "rb")
        data = np.array(pickle.load(file))
        file.close()
        return data