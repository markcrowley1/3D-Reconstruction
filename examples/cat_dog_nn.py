
import pickle
import numpy as np
from keras import models, layers, utils

def main():
    # Read data in for training
    pickle_in = open("examples/x.pickle", "rb")
    x = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("examples/y.pickle", "rb")
    y = pickle.load(pickle_in)
    pickle_in.close()

    # Scale Data
    X = x/255.0
    y = np.array(y)

    print(X.shape[1:])
    print(y[1])

    # Define model
    model = models.Sequential()

    # Input layer
    model.add(layers.Conv2D(64, (3,3), activation="relu" , input_shape=X.shape[1:]))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    # Hidden Layer
    model.add(layers.Conv2D(64, (3,3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    # Output layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.summary()

    model.compile(loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])
    
    model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)
    model.save("examples/cat_dog_classifier.keras")

if __name__ == "__main__":
    main()