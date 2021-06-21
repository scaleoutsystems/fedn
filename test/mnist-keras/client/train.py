from __future__ import print_function
import sys
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as krm
import numpy as np
import pickle
import yaml
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from data.read_data import read_data
import os


def train(model,data,settings):
    print("-- RUNNING TRAINING --", flush=True)

    # We are caching the partition in the container home dir so that
    # the same training subset is used for each iteration for a client.
    try:
        x_train = np.load('/tmp/local_dataset/x_train.npz')
        y_train = np.load('/tmp/local_dataset/y_train.npz')
    except:
        (x_train, y_train, classes) = read_data(data,
                                                nr_examples=settings['training_samples'],
                                                trainset=True)

        try:
            os.mkdir('/tmp/local_dataset')
            np.save('/tmp/local_dataset/x_train.npz',x_train)
            np.save('/tmp/local_dataset/y_train.npz',y_train)

        except:
            pass

    model.fit(x_train, y_train, batch_size=settings['batch_size'], epochs=settings['epochs'], verbose=1)

    print("-- TRAINING COMPLETED --", flush=True)
    return model

if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)

    from fedn.utils.kerashelper import KerasHelper
    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])

    from models.mnist_model import create_seed_model
    model = create_seed_model()
    model.set_weights(weights)

    model = train(model,'../data/mnist.npz',settings)
    helper.save_model(model.get_weights(),sys.argv[2])
