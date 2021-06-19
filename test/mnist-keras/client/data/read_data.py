import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras

def read_data(path, nr_examples=1000,trainset=True):
    """ Helper function to read and preprocess data for training with Keras. """

    pack = np.load(path)

    if trainset:
        X = pack['x_train']
        y = pack['y_train']
    else:
        X = pack['x_test']
        y = pack['y_test']

    print("X shape: ", X.shape, ", y shape: ", y.shape)

    sample_fraction = float(nr_examples)/len(y)

    # The entire dataset is 60k images, we can subsample here for quicker testing.
    if sample_fraction < 1.0:
        _, X, _, y = train_test_split(X, y, test_size=sample_fraction)
    classes = range(10)

    # Input image dimensions
    img_rows, img_cols = 28, 28

    ## The data, split between train and test sets
    #X = X.reshape(X.shape[0], img_rows, img_cols, 1)
    X = X.astype('float32')
    X = np.expand_dims(X,-1)
    X /= 255
    y = tf.keras.utils.to_categorical(y, len(classes))
    return  (X, y, classes)
