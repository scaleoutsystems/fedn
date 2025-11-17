import os
import sys

import numpy as np
from data import load_data
from model import load_parameters

from scaleout.utils.helpers.helpers import save_metrics

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def validate(in_model_path, out_json_path, data_path=None):
    """Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    # Load data
    x_train, y_train = load_data(data_path)
    x_test, y_test = load_data(data_path, is_train=False)

    # Load model
    model = load_parameters(in_model_path)

    # Evaluate
    model_score = model.evaluate(x_train, y_train)
    model_score_test = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    # JSON schema
    report = {
        "training_loss": model_score[0],
        "training_accuracy": model_score[1],
        "test_loss": model_score_test[0],
        "test_accuracy": model_score_test[1],
    }

    # Save JSON
    save_metrics(report, out_json_path)


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
