import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import yaml


def read_data(filename):
    """
    Helper function to read and preprocess data for training with Keras.
    :return: model
    """
    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise e

    print("-- START READING DATA --")
    # The entire dataset is 50k reviews, we can subsample here for quicker testing.
    data_input = np.array(pd.read_csv(filename))
    print(data_input.shape)

    x_train = data_input[:, 1:-1]
    y_train = data_input[:, -1]
    _, X, _, y = train_test_split(x_train, y_train, test_size=settings['test_size'])
    return X, y
