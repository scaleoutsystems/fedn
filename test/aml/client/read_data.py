import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

import keras
import numpy
num_classes = 15

def pre_process(x_data, y_data):

    y_data = keras.utils.to_categorical(y_data, num_classes)
    x_data = x_data.astype('float32')
    x_data /= 255
    return x_data, y_data

def read_data(filename, sample_fraction=1.0):
    """ Helper function to read and preprocess data for training with Keras. """

    import_folder = 'pickled_data_trainvaltest'

    X = pickle.load(open(filename + "/pickled_data_trainvaltest/x_train.p", "rb"))
    y = pickle.load(open(filename + "/pickled_data_trainvaltest/y_train.p", "rb"))
    X, y = pre_process(X, y)

    # The entire dataset is 60k images, we can subsample here for quicker testing. 
    if sample_fraction < 1.0:
        foo, X, bar, y = train_test_split(X, y, test_size=sample_fraction)
    classes = range(num_classes)


    return  (X, y, classes)


