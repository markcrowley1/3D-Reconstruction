"""
Description:
    Train simple network on mnist dataset
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def view_data(x):
    print(x.shape)
    print(type(x))
    plt.imshow(x[1], cmap = plt.cm.binary)
    plt.show()
    return

def save_and_load_model(model: tf.keras.models.Sequential, x_test: np.ndarray):
    # Save and load model to make prediction
    model.save("./mnist_example.model")
    new_model = tf.keras.models.load_model("./mnist_example.model")
    predictions = new_model.predict([x_test])
    print(np.argmax(predictions[0]))

def main():
    """Multilayer Perceptron network for recognizing mnist digits"""
    # Load dataset
    dataset = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    # Normalise training data
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # Create sequential model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    
    # Train model
    model.fit(x_train, y_train, epochs=3)

    # Test model and print results
    val_loss, val_accuracy = model.evaluate(x_test, y_test)
    print(val_loss, val_accuracy)


if __name__ == "__main__":
    main()