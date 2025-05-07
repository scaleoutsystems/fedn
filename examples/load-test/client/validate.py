# /bin/python
import os
import sys
import time

import numpy as np
from model import load_model

from fedn.utils.helpers.helpers import save_metrics

HELPER_MODULE = "numpyhelper"
ARRAY_SIZE = 1000000


def validate(in_model_path, out_json_path):
    """Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    tic = time.time()
    weights = load_model(in_model_path)
    toc = time.time()

    # size on disk in bytes of in_model_path
    size_on_disk = os.path.getsize(in_model_path)

    # JSON schema
    report = {
        "memory_size_bytes": np.array(weights).nbytes,
        "load_time_seconds": toc - tic,
        "size_on_disk_bytes": size_on_disk,
    }

    # Save JSON
    save_metrics(report, out_json_path)


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
