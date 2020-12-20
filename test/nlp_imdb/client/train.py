from __future__ import print_function
import sys
import tensorflow as tf
from read_data import read_data
import yaml

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train(model, data, settings):
    """
    Helper function to train the model
    :return: model
    """
    print("-- RUNNING TRAINING --")

    x_train, y_train = read_data(data)
    model.fit(x_train, y_train, epochs=settings['epochs'], batch_size=settings['batch_size'], verbose=True)

    print("-- TRAINING COMPLETED --")
    return model


if __name__ == '__main__':

    from fedn.utils.kerasweights import KerasWeightsHelper
    from models.imdb_model import create_seed_model

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise e

    helper = KerasWeightsHelper()
    weights = helper.load_model(sys.argv[1])
    model = create_seed_model()
    model.set_weights(weights)
    model = train(model, '../data/train.csv', settings)
    helper.save_model(model.get_weights(), sys.argv[2])