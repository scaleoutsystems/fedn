# /bin/python
import sys
import time

import numpy as np
from model import load_model, save_model

from scaleout.utils.helpers.helpers import save_metadata

HELPER_MODULE = "numpyhelper"
ARRAY_SIZE = 10000


def train(in_model_path, out_model_path):
    """ Train model.

    """
    # Load model
    weights = load_model(in_model_path)

    # Train
    time.sleep(np.random.randint(4, 15))

    # Metadata needed for aggregation server side
    metadata = {
        "num_examples": ARRAY_SIZE,
    }

    # Save JSON metadata file
    save_metadata(metadata, out_model_path)

    # Save model update
    save_model(weights, out_model_path)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
