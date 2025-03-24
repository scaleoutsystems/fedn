import os
import sys

import torch
from data import load_data
from model import load_parameters

from fedn.utils.helpers.helpers import save_metrics

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def predict(in_model_path, out_json_path, data_path=None):
    """Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the predict output to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    # Load data
    x_test, y_test = load_data(data_path, is_train=False)
    # Load model
    model = load_parameters(in_model_path)
    model.eval()

    # Predict
    with torch.no_grad():
        y_pred = model(x_test)
        y_pred = torch.argmax(y_pred, dim=1)

    result = {"predicted_class": y_pred.tolist()}

    save_metrics(result, out_json_path)


if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2])
