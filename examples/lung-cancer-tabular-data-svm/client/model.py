import collections
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss

from fedn.utils.helpers.helpers import get_helper


HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


def compile_model():
    
    # Define the SVM model
   
    model = SGDClassifier(max_iter=10, tol=1e-3, loss='log_loss')
    n_features = 15
    dummy_X = np.zeros((2, n_features))  # A single sample with all zeros
    dummy_y = np.array([0,1])  # A dummy label 
    model.fit(dummy_X, dummy_y)

    return model


def save_parameters(model, out_path):
    """Save model parameters to file.

    :param model: The model to serialize.
    :type model: keras.model.Sequential
    :param out_path: The path to save the model to.
    :type out_path: str
    """

    weights = model.coef_
    intercept = model.intercept_

    parameters_np = [weights, intercept]

    #print('parameters_np: ', parameters_np)

    helper.save(parameters_np, out_path)

def load_parameters(model_path):
    """Load model parameters from file and populate model.

    :param model_


    path: The path to load from.
    :type model_path: str
    :return: The loaded model.
    :rtype: keras.model.Sequential
    

    """
    model = compile_model()
    parameters_np = helper.load(model_path)
    model.coef_ = parameters_np[0]
    model.intercept_ = parameters_np[1]

    return model

def init_seed(out_path="../seed.npz"):
    """Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    model = compile_model()
    save_parameters(model, out_path)

if __name__ == "__main__":
    init_seed("../seed.npz")
