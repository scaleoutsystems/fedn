import numpy
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler, LabelEncoder
import keras
import sys
import numpy as np
import scipy
import scipy.io
# import sys
# sys.setrecursionlimit(10000)
from keras.utils import to_categorical
import yaml


def read_data(filename):


    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise e
    """ Helper function to read and preprocess data for training with Keras. """

    """ Read the prepared and normlized data, where the features number is 36 and the output is 10 classes """

    pkd = np.array(pd.read_csv(filename))
    x = pkd[:, 1:37]
    y = pkd[:, 37:]
    _, X, _, Y  = train_test_split(x, y,test_size=settings['test_size'])
    """reshaped the input data for LSTM model """
    X = X.reshape(X.shape[0], 1, X.shape[1])

    return X, Y


