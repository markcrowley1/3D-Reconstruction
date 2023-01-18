import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    dataset = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    print(x_train.shape)
    print(type(x_train))
    plt.imshow(x_train[1], cmap = plt.cm.binary)
    plt.show()

if __name__ == "__main__":
    main()