import numpy
import pandas as pd
from sklearn.model_selection import train_test_split

import keras
import numpy

def read_data(filename, sample_fraction=1.0):
    """ Helper function to read and preprocess data for training with Keras. """

    data = numpy.array(pd.read_csv(filename))
    X = data[:, :-1]
    y = data[:, -1]

    if sample_fraction < 1.0:
        foo, X, bar, y = train_test_split(X, y, test_size=sample_fraction)

    X = X.reshape(X.shape[0], X.shape[1], 1)
    X = X.astype('float32')
    X /= 255
    return  (X, y)

