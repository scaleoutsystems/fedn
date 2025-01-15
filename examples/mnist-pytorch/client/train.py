import math
import os
import sys

import torch
from data import load_data
from model import load_parameters, save_parameters

from fedn.utils.helpers.helpers import save_metadata

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def train(in_model_path, out_model_path, data_path=None, batch_size=32, epochs=1, lr=0.01):
    """Complete a model update.

    Load model paramters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated paramters
    to out_model_path (picked up by the FEDn client).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param epochs: The number of epochs to train.
    :type epochs: int
    :param lr: The learning rate to use.
    :type lr: float
    """
    # Load data
    x_train, y_train = load_data(data_path)

    # Load parmeters and initialize model
    model = load_parameters(in_model_path)

    # Train
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    n_batches = int(math.ceil(len(x_train) / batch_size))
    criterion = torch.nn.NLLLoss()
    for e in range(epochs):  # epoch loop
        for b in range(n_batches):  # batch loop
            # Retrieve current batch
            batch_x = x_train[b * batch_size : (b + 1) * batch_size]
            batch_y = y_train[b * batch_size : (b + 1) * batch_size]
            # Train on batch
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            # Log
            if b % 100 == 0:
                print(f"Epoch {e}/{epochs-1} | Batch: {b}/{n_batches-1} | Loss: {loss.item()}")

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": len(x_train),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
