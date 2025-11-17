import sys
from model import load_parameters, save_parameters
from data import load_data
from scaleout.utils.helpers.helpers import save_metadata
import os

# Get the list of all files and directories

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def train(in_model_path, out_model_path, data_path=None, batch_size=16, epochs=1, lr=0.01):
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
    data, length = load_data(data_path, step="train")
    # Load parmeters and initialize model
    model = load_parameters(in_model_path)
    # Train
    model.train(data=data, epochs=epochs, imgsz=640, batch=batch_size, lr0=lr, warmup_epochs=0, optimizer="Adam")


    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": length,
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
