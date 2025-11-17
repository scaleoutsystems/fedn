import os
import sys

import torch
from data import load_data
from model import load_client_model, save_client_model
from torch import optim

from scaleoututil.logging import FednLogger
from scaleoututil.helpers.helpers import get_helper

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)

HELPER_MODULE = "splitlearninghelper"
helper = get_helper(HELPER_MODULE)

seed = 42
torch.manual_seed(seed)


def backward_pass(gradient_path, client_id):
    """Loads gradients and extracts the relevant gradients to update the client model for the given client.

    param gradient_path: Path to the gradients file.
    :type gradient_path: str
    param client_id: ID of the client to update.
    :type client_id: str
    """
    FednLogger().info(f"Performing backward pass for client {client_id}")

    x_train = load_data(data_path=None, is_train=True)
    num_local_features = x_train.shape[1]

    client_model = load_client_model(client_id, num_local_features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_model.to(device)

    client_optimizer = optim.Adam(client_model.parameters(), lr=0.01)
    client_optimizer.zero_grad()

    # recomputing the computational graph
    embedding = client_model(x_train)

    gradients = helper.load(gradient_path)

    local_gradients = gradients[client_id]
    local_gradients = torch.tensor(local_gradients, dtype=torch.float32)

    embedding.backward(local_gradients)

    client_optimizer.step()

    # save the updated model
    save_client_model(client_model, client_id)


if __name__ == "__main__":
    backward_pass(sys.argv[1], sys.argv[2])
