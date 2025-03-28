# /bin/python

import os

import numpy as np

from fedn.utils.helpers.helpers import get_helper

HELPER_MODULE = "numpyhelper"
ARRAY_SIZE_FACTOR = int(os.environ.get("ARRAY_SIZE_FACTOR", 1))
# 144 MB * ARRAY_SIZE_FACTOR
ARRAY_SIZE = 20000000 * ARRAY_SIZE_FACTOR


def save_model(weights, out_path):
    """Save model to disk.

    :param model: The model to save.
    :type model: torch.nn.Module
    :param out_path: The path to save to.
    :type out_path: str
    """
    helper = get_helper(HELPER_MODULE)
    helper.save(weights, out_path)


def load_model(model_path):
    """Load model from disk.

    param model_path: The path to load from.
    :type model_path: str
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    helper = get_helper(HELPER_MODULE)
    weights = helper.load(model_path)
    return weights


def init_seed(out_path="seed.npz"):
    """Initialize seed model.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    # Init and save
    weights = [np.random.rand(1, ARRAY_SIZE)]
    save_model(weights, out_path)


if __name__ == "__main__":
    init_seed("../seed.npz")
