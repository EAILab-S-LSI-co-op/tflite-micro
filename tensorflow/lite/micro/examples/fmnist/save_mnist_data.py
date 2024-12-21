import tensorflow as tf
from tensorflow import keras
import numpy as np

def load_mnist_data():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Reshape images to be suitable for the neural network
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_mnist_data()
x_train.astype(np.float32).tofile('x_train.bin')
y_train.astype(np.int32).tofile('y_train.bin')
x_test.astype(np.float32).tofile('x_test.bin')
y_test.astype(np.int32).tofile('y_test.bin')