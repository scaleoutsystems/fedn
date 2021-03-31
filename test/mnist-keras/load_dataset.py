import os
import numpy as np
import tensorflow as tf
import urllib.request


if __name__ == '__main__':

    path = 'data'
    if not os.path.exists(path):
        os.mkdir(path)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    np.save('data/x_train', x_train)
    np.save('data/y_train', y_train)
    np.save('data/x_test', x_test)
    np.save('data/y_test', y_test)

    #url = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
    #urllib.request.urlretrieve(url, 'data/mnist.npz')