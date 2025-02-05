import os
import struct
import tempfile

import numpy as np

from fedn.common.log_config import logger
from fedn.utils.helpers.helperbase import HelperBase


class Helper(HelperBase):
    """FEDn helper class for android json model weights."""

    def __init__(self):
        """Initialize helper."""
        self.name = "androidhelper"
        super().__init__()

    # function to calculate an incremental weighted average of the weights

    def increment_average(self, model, model_next, num_examples, total_examples):
        """Incremental weighted average of model weights.

        :param model: Current model weights.
        :type model: list of numpy arrays.
        :param model_next: New model weights.
        :type model_next: list of numpy arrays.
        :param num_examples: Number of examples in new model.
        :type num_examples: int
        :param total_examples: Total number of examples.
        :type total_examples: int
        :return: Incremental weighted average of model weights.
        :rtype: list of numpy arrays.
        """
        # Incremental weighted average
        w = num_examples / total_examples

        return (1 - w) * model + w * model_next

    # function to calculate an incremental weighted average of the weights using numpy.add
    def increment_average_add(self, model, model_next, num_examples, total_examples):
        """Incremental weighted average of model weights.

        :param model: Current model weights.
        :type model: list of numpy arrays.
        :param model_next: New model weights.
        :type model_next: list of numpy arrays.
        :param num_examples: Number of examples in new model.
        :type num_examples: int
        :param total_examples: Total number of examples.
        :type total_examples: int
        :return: Incremental weighted average of model weights.
        :rtype: list of numpy arrays.
        """
        # Incremental weighted average
        w = np.add(
            model,
            num_examples * (np.array(model_next) - np.array(model)) / total_examples,
        )
        return w

    def save(self, weights, path=None):
        """Serialize weights to file. The serialized model must be a single binary object.

        :param weights: weights in json format.
        :param path: Path to file.
        :return: Path to file.
        """
        if not path:
            path = self.get_tmp_path()

        byte_array = struct.pack("f" * len(weights), *weights)
        with open(path, "wb") as file:
            file.write(byte_array)

        return path

    def load(self, fh):
        """Load weights from file or filelike.

        :param fh: file path, filehandle, filelike.
        :return: List of weights in json format.
        """
        logger.debug("in android helper load")
        if isinstance(fh, str):
            with open(fh, "rb") as file:
                byte_data = file.read()
        else:
            byte_data = fh.read()

        weights = np.array(struct.unpack(f"{len(byte_data) // 4}f", byte_data))

        return weights

    def get_tmp_path(self):
        """Return a temporary output path compatible with save_model, load_model.

        :return: Path to file.
        """
        fd, path = tempfile.mkstemp(suffix="")
        os.close(fd)
        return path
