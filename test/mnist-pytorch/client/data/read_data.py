from torch.utils.data import Subset

import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def read_data(trainset=True, nr_examples=1000,bias=0.7, data_path='../data/mnist.npz'):
    """ Helper function to read and preprocess data for training with Keras. """

    pack = np.load(data_path)

    if trainset:
        X = pack['x_train']
        y = pack['y_train']
    else:
        X = pack['x_test']
        y = pack['y_test']

    X = X.astype('float32')
    y = y.astype('int64')

    X = np.expand_dims(X, 1)
    X /= 255
    tensor_x = torch.Tensor(X)  # transform to torch tensor
    tensor_y = torch.from_numpy(y)
    dataset = TensorDataset(tensor_x, tensor_y)  # create traindatset
    sample = np.random.choice(np.arange(len(dataset)),nr_examples,replace=False)
    dataset = Subset(dataset=dataset, indices=sample)

    return dataset



