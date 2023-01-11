"""
Description:
    Checking that tensorflow working with gpu
"""

import tensorflow as tf

def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    tf.debugging.set_log_device_placement(True)

    # Create some tensors
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

    print(c)

if __name__ == "__main__":
    main()