import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy

def read_data(filename, nr_examples=1000, batch_size=100):
    """ Helper function to read and preprocess data for training with pytorch. """

    print("inside read data: filename: ", filename)

    data = numpy.array(pd.read_csv(filename))
    print("data: ", data.shape)
    X = data[:, 1::]
    y = data[:, 0]

    sample_fraction = float(nr_examples)/len(y)  

    # The entire dataset is 60k images, we can subsample here for quicker testing. 
    if sample_fraction < 1.0:
        _, X, _, y = train_test_split(X, y, test_size=sample_fraction)
    classes = range(10)

    # Input image dimensions
    img_rows, img_cols = 28, 28

    # The data, split between train and test sets
    X = X.reshape(X.shape[0], 1, img_rows, img_cols)
    X = X.astype('float32')
    X /= 255
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader

