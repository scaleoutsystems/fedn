import os
import sys

import torch
import torch.optim as optim
from data import load_data
from model import load_parameters, save_parameters
from torch.utils.data import DataLoader, TensorDataset
from fedn.utils.helpers.helpers import save_metadata


def train(in_model_path, out_model_path, data_path=None, batch_size=10, epochs=1):
    
    # Load data
    X_train, y_train = load_data(data_path)


    # Load model
    model = load_parameters(in_model_path)

    model.fit(X_train, y_train)

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": len(X_train),
        "batch_size": batch_size,
        "epochs": epochs,
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
