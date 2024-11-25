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

from fedn.utils.helpers.helpers import get_helper


HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


def compile_model():
    
    # Define the Neural Network
    class LungCancerNN(nn.Module):


        def __init__(self):
            super(LungCancerNN, self).__init__()
            self.fc1 = nn.Linear(15, 5)  # 15 input features
            self.fc2 = nn.Linear(5, 5)
            self.fc3 = nn.Linear(5, 1)   # Output layer with 1 neuron (binary classification)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x
    
    # Initialize the network, loss function, and optimizer
    model = LungCancerNN()
    
    return model


def save_parameters(model, out_path):
    """Save model parameters to file.

    :param model: The model to serialize.
    :type model: keras.model.Sequential
    :param out_path: The path to save the model to.
    :type out_path: str
    """
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    

    helper.save(parameters_np, out_path)

def load_parameters(model_path):
    """Load model parameters from file and populate model.

    :param model_path: The path to load from.
    :type model_path: str
    :return: The loaded model.
    :rtype: keras.model.Sequential
    """
    model = compile_model()
    parameters_np = helper.load(model_path)


    params_dict = zip(model.state_dict().keys(), parameters_np)
    state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
    model.load_state_dict(state_dict, strict=True)
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
