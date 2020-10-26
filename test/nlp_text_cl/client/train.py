from __future__ import print_function
import sys
import keras
import tensorflow as tf
from read_data import read_data
import tensorflow.keras.models as krm

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train(model, data):
    print("-- RUNNING TRAINING --")
    BATCH_SIZE = 5
    EPOCHS = 3

    (x_train, x_test, y_train, y_test) = read_data(data)
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=True)

    print("-- TRAINING COMPLETED --")
    return model


if __name__ == '__main__':
    # print('TF', tf.__version__)
    # print('KERAS', keras.__version__)
    model = krm.load_model(sys.argv[1])
    model = train(model, '../data/train.csv')
    model.save(sys.argv[2])


