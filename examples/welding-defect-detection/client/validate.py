import os
import sys

from model import load_parameters
from data import load_data
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
    test_data_yaml, test_data_length = load_data(data_path, step="test")
    model = load_parameters(in_model_path)
    validation_results = model.val(data=test_data_yaml)


    # JSON schema
    report = {
        "map50-95": float(validation_results.box.map),  # map50-95
        "map50": float(validation_results.box.map50),  # map50
        "map75": float(validation_results.box.map75),  # map75
    }
    # Save JSON
    save_metrics(report, out_json_path)


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
