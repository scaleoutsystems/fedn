import os
import sys

import numpy as np
import torch
from model import load_client_model, save_client_model
from torch import optim

from fedn.common.log_config import logger
from fedn.utils.helpers.helpers import get_helper, save_metadata

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)

HELPER_MODULE = "splitlearninghelper"
helper = get_helper(HELPER_MODULE)


def backward_pass(gradient_path, client_id):
    """Load gradients from in_gradients_path, load the embeddings, and perform a backward pass to update 
    the parameters of the client model. Save the updated model to out_model_path.
    """
    # load client model with parameters 
    client_model = load_client_model(client_id)
    logger.info(f"Client model loaded from {client_id}")

    # instantiate optimizer
    client_optimizer = optim.Adam(client_model.parameters(), lr=0.01)
    client_optimizer.zero_grad()

    # load local embedding from previous forward pass
    logger.info(f"Loading embedding from {client_id}")
    try:
        npz_file = np.load(f"{abs_path}/embeddings/embeddings_{client_id}.npz")
        embedding = next(iter(npz_file.values()))
    except FileNotFoundError:
        raise FileNotFoundError(f"Embedding file {client_id} not found")

    # transform to tensor
    embedding = torch.tensor(embedding, dtype=torch.float32, requires_grad=True)

    # load gradients
    gradients = helper.load(gradient_path)
    logger.info(f"Gradients loaded from {gradient_path}")

    local_gradients = gradients[client_id]
    local_gradients = torch.tensor(local_gradients, dtype=torch.float32, requires_grad=True)

    # perform backward pass
    embedding.backward(local_gradients)
    client_optimizer.step()

    # save updated client model locally
    save_client_model(client_model, client_id)

    logger.info(f"Updated client model saved to {abs_path}/local_models/{client_id}.pth")

if __name__ == "__main__":
    backward_pass(sys.argv[1], sys.argv[2])
