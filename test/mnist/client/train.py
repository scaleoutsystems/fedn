from __future__ import print_function
import sys
import tensorflow as tf
import tensorflow.keras as keras 
import tensorflow.keras.models as krm

import pickle
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from read_data import read_data


def train(model,data,sample_fraction):
    print("-- RUNNING TRAINING --")

    batch_size = 32
    num_classes = 10
    epochs = 1

    # Input image dimensions
    img_rows, img_cols = 28, 28

    # The data, split between train and test sets
    (x_train, y_train, classes) = read_data(data,sample_fraction=sample_fraction)
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    print("-- TRAINING COMPLETED --")
    return model

if __name__ == '__main__':
    model = krm.load_model(sys.argv[1])
    model = train(model,'../data/train.csv',sample_fraction=0.01)
    model.save(sys.argv[2])


