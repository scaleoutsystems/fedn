import os

import torch
from torch import nn

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


class ClientModel(nn.Module):
    """Client-side model"""

    def __init__(self, input_features):
        super(ClientModel, self).__init__()
        self.fc1 = nn.Linear(input_features, 8)
        self.fc2 = nn.Linear(8, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


def compile_model(num_local_features):
    """Build the client model.

    param num_local_features: Number of features in the local dataset.
    :type num_local_features: int
    :return: The client model.
    :rtype: ClientModel
    """
    model = ClientModel(num_local_features)
    return model


def save_client_model(model, client_id):
    """Save the client model to the local_models directory (saves model locally).

    param model: The client model.
    :type model: ClientModel
    param client_id: ID of the client to save the model.
    :type client_id: str
    """
    if not os.path.exists(f"{abs_path}/local_models"):
        os.makedirs(f"{abs_path}/local_models")
    torch.save(model.state_dict(), f"{abs_path}/local_models/{client_id}.pth")


def load_client_model(client_id, num_local_features):
    """Load the client model from the local_models directory.

    param client_id: ID of the client to load the model.
    :type client_id: str
    param num_local_features: Number of features in the local dataset.
    :type num_local_features: int
    :return: The client model.
    :rtype: ClientModel
    """
    model = compile_model(num_local_features)
    model.load_state_dict(torch.load(f"{abs_path}/local_models/{client_id}.pth", weights_only=True))
    return model
