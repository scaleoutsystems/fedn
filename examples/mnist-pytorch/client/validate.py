import os
import sys

import torch
from data import load_data
from model import load_parameters
from torch.utils.data import DataLoader
from fedn.utils.helpers.helpers import save_metrics

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def validate(in_model_path, out_json_path, data_path=None, batch_size=32):
    """ Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """

    chunk_dataset = load_data(data_path)
    chunk_loader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True)

    chunk_dataset_test = load_data(data_path,is_train=False)
    chunk_loader_test = DataLoader(chunk_dataset_test, batch_size=batch_size, shuffle=True)

    # Load model
    model = load_parameters(in_model_path)
    model.eval()

    # Evaluate
    criterion = torch.nn.NLLLoss()
    with torch.no_grad():

        for b, (x_train, y_train) in enumerate(chunk_loader):
            train_out = model(x_train)
            training_loss = criterion(train_out, y_train)
            training_accuracy = torch.sum(torch.argmax(train_out, dim=1) == y_train) / len(train_out)

        for b, (x_test, y_test) in enumerate(chunk_loader):
            test_out = model(x_test)
            test_loss = criterion(test_out, y_test)
            test_accuracy = torch.sum(torch.argmax(test_out, dim=1) == y_test) / len(test_out)

    # JSON schema
    report = {
        "training_loss": training_loss.item(),
        "training_accuracy": training_accuracy.item(),
        "test_loss": test_loss.item(),
        "test_accuracy": test_accuracy.item(),
    }

    # Save JSON
    save_metrics(report, out_json_path)


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
