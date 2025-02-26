import os

import torch
from torch import nn

from fedn.utils.helpers.helpers import get_helper

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)

HELPER_MODULE = "splitlearninghelper"
helper = get_helper(HELPER_MODULE)


class ClientModel(nn.Module):
    """Client-side model"""

    def __init__(self, input_features):
        super(ClientModel, self).__init__()
        self.fc1 = nn.Linear(input_features, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return x

def compile_model(num_local_features=2):
    """Build the client model"""
    model = ClientModel(num_local_features)
    return model

def save_client_model(model, client_id):
    """Save the client model to the local_models directory (saves model locally)"""
    if not os.path.exists(f"{abs_path}/local_models"):
        os.makedirs(f"{abs_path}/local_models")
    torch.save(model.state_dict(), f"{abs_path}/local_models/{client_id}.pth")

def load_client_model(client_id, num_local_features=2):
    """Load the client model from the local_models directory"""
    model = compile_model(num_local_features)
    model.load_state_dict(torch.load(f"{abs_path}/local_models/{client_id}.pth", weights_only=True))
    return model
