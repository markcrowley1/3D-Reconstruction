"""
Description:
    Run inference on image specified as argument.
"""

import sys
import pickle
import numpy as np
from keras import models
from tensorflow_graphics.nn import loss

def load_images(filename: str) -> np.ndarray:
    # Load in images
    file = open(filename, "rb")
    imgs = np.array(pickle.load(file))
    file.close()
    # Normalise images
    imgs = imgs/255.0
    return imgs

def main():
    filename = sys.argv[1]
    images = load_images(filename)
    
    model: models.Sequential = models.load_model(
        "overfit_psg.keras",
        custom_objects={'evaluate': loss.chamfer_distance.evaluate}
    )
    output = model.predict(images[1:])

    pickle_out = open("unseen_output.pickle", "wb")
    pickle.dump(output, pickle_out)
    pickle_out.close()

if __name__ == "__main__":
    main()