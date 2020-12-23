from __future__ import print_function
import sys
import keras
from keras.models import Sequential
import tensorflow as tf
import random
import yaml


import pickle
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from read_data import read_data
import os   


def train(model,data, settings):
    print("-- RUNNING TRAINING --", flush=True)

    x_train, y_train = read_data(data)

    model.fit(x_train, y_train, epochs=settings['epochs'], batch_size=settings['batch_size'], verbose=True)

    print("-- TRAINING COMPLETED --", flush=True)
    return model

if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise e
    from fedn.utils.kerassequential import KerasSequentialHelper
    helper = KerasSequentialHelper()
    model = helper.load_model(sys.argv[1])
    model = train(model, '../data/train.csv', settings)
    helper.save_model(model, sys.argv[2])



