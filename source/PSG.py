"""
Description:
    Class handles point set generation from input image.
"""
import cv2
import numpy as np
from keras import models
from tensorflow_graphics.nn import loss

class PSG():
    def __init__(self, model: str):
        """ Load model """
        self.model: models.Sequential = models.load_model(
            model,
            custom_objects={'evaluate': loss.chamfer_distance.evaluate}
        )
        self.input_height = self.model.input_shape[1]
        self.input_width  = self.model.input_shape[2]
            
    def generate_pointset(self, img: str):
        """ Preprocess Image and Predict Point Cloud """
        # Preprocess
        img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (self.input_height, self.input_width))
        img_array = np.array(img_array).reshape(-1, self.input_height, self.input_width, 1)
        img_array = img_array/255.0
        # Predict
        return self.model.predict(img_array)[0]