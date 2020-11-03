from __future__ import print_function
import sys
import tensorflow as tf
import tensorflow.keras as keras 
import tensorflow.keras.models as krm

import pickle
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from read_data import read_data
import os   


def train(model,data,sample_fraction):
    print("-- RUNNING TRAINING --", flush=True)

    batch_size = 32
    num_classes = 10
    epochs = 1



    # The data, split between train and test sets. We are caching the partition in 
    # the container home dir so that the same training subset is used for 
    # each iteration. 
    try:
        with open('/app/data/x.pyb','rb') as fh:
            x_train=pickle.loads(fh.read())
        with open('/app/data/y.pyb','rb') as fh:
            y_train=pickle.loads(fh.read())
        with open('/app/data/classes.pyb','rb') as fh:
            classes=pickle.loads(fh.read())
    except:
        (x_train, y_train, classes) = read_data(data,sample_fraction=sample_fraction)

    try:
        os.mkdir('/app/data')
        with open('/app/data/x.pyb','wb') as fh:
            fh.write(pickle.dumps(x_train))
        with open('/app/data/y.pyb','wb') as fh:
            fh.write(pickle.dumps(y_train))
        with open('/app/data/classes.pyb','wb') as fh:
            fh.write(pickle.dumps(classes))
    except:
        pass


    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    print("-- TRAINING COMPLETED --", flush=True)
    return model

if __name__ == '__main__':
    model = krm.load_model(sys.argv[1])
    model = train(model,'../data/pickled_data_trainvaltest',sample_fraction=0.1)
    model.save(sys.argv[2])


