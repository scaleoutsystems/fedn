from __future__ import print_function
import sys
import tensorflow as tf
from read_data import read_data
import tensorflow.keras.models as krm
import random
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train(model, data):
    print("-- RUNNING TRAINING --")
    batch_size = 64
    epochs = 1

    (x_train, x_test, y_train, y_test, tokenizer) = read_data(data)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=True)

    print("-- TRAINING COMPLETED --")
    return model


def selectdata():
    # number of training dataset (clients)
    data_number = 3
    data_path = '../data/data2/train_%g.csv'%random.randrange(1, data_number)
    print('DATA:', data_path)
    return data_path


if __name__ == '__main__':
    # print('TF', tf.__version__)
    # print('KERAS', keras.__version__)
    model = krm.load_model(sys.argv[1])
    # model = train(model, '../data/train.csv')
    model = train(model, selectdata())
    model.save(sys.argv[2])