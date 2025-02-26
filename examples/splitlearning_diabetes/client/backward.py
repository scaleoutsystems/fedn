import os
import sys

import numpy as np
import torch
from data import load_data
from model import load_client_model, save_client_model
from torch import optim

from fedn.common.log_config import logger
from fedn.utils.helpers.helpers import get_helper

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)

HELPER_MODULE = "splitlearninghelper"
helper = get_helper(HELPER_MODULE)


def get_computational_graph(client_model, data_path=None):
    x_train = load_data(data_path, is_train=True)
    embedding = client_model(x_train)
    return embedding


def backward_pass(gradient_path, client_id):
    """Load gradients from in_gradients_path, load the embeddings, and perform a backward pass to update
    the parameters of the client model. Save the updated model to out_model_path.
    """
    logger.info(f"Performing backward pass for client {client_id}")

    x_train = load_data(data_path=None, is_train=True)
    num_local_features = x_train.shape[1]

    client_model = load_client_model(client_id, num_local_features)

    # instantiate optimizer
    client_optimizer = optim.Adam(client_model.parameters(), lr=0.01)
    client_optimizer.zero_grad()

    embedding = client_model(x_train)

    # load gradients
    gradients = helper.load(gradient_path)
    logger.info("Gradients loaded")

    local_gradients = gradients[client_id]
    local_gradients = torch.tensor(local_gradients, dtype=torch.float32)

    embedding.backward(local_gradients)

    logger.info("backward pass performed")

    client_optimizer.step()

    save_client_model(client_model, client_id)


if __name__ == "__main__":
    backward_pass(sys.argv[1], sys.argv[2])
