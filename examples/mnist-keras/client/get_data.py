import os
from math import floor

import numpy as np
import tensorflow as tf


def splitset(dataset, parts):
    n = dataset.shape[0]
    local_n = floor(n/parts)
    result = []
    for i in range(parts):
        result.append(dataset[i*local_n: (i+1)*local_n])
    return np.array(result)


def split(dataset='data/mnist.npz', outdir='data', n_splits=2):
    # Load and convert to dict
    package = np.load(dataset)
    data = {}
    for key, val in package.items():
        data[key] = splitset(val, n_splits)

    # Make dir if necessary
    if not os.path.exists(f'{outdir}/clients'):
        os.mkdir(f'{outdir}/clients')

    # Make splits
    for i in range(n_splits):
        subdir = f'{outdir}/clients/{str(i+1)}'
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        np.savez(f'{subdir}/mnist.npz',
                 x_train=data['x_train'][i],
                 y_train=data['y_train'][i],
                 x_test=data['x_test'][i],
                 y_test=data['y_test'][i])


def get_data(out_dir='data'):
    # Make dir if necessary
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Download data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    np.savez(f'{out_dir}/mnist.npz', x_train=x_train,
             y_train=y_train, x_test=x_test, y_test=y_test)


if __name__ == '__main__':
    get_data()
    split()
