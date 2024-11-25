import json
import os
import sys

import numpy as np
from data import load_data
from model import load_parameters

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def predict(in_model_path, out_json_path, data_path=None):
    # Using test data for prediction but another dataset could be loaded
    x_test, _ = load_data(data_path, is_train=False)

    # Load model
    model = load_parameters(in_model_path)

    # Predict
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    # Save JSON
    with open(out_json_path, "w") as fh:
        fh.write(json.dumps({"predictions": y_pred.tolist()}))


if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2])
