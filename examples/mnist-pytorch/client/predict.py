import os
import sys

import torch
from data import load_data
from model import load_parameters

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def predict(in_model_path, out_artifact_path, data_path=None):
    """Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_artifact_path: The path to save the predict output to.
    :type out_artifact_path: str
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
    # Save prediction to file/artifact, the artifact will be uploaded to the object store by the client
    torch.save(y_pred, out_artifact_path)


if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2])
