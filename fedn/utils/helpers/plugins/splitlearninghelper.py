import os
import tempfile

import numpy as np

# import torch
from fedn.common.log_config import logger
from fedn.utils.helpers.helperbase import HelperBase


class Helper(HelperBase):
    """FEDn helper class for models weights/parameters that can be transformed to numpy ndarrays."""

    def __init__(self):
        """Initialize helper."""
        super().__init__()
        self.name = "splitlearninghelper"

    def increment_average(self, embedding1, embedding2):
        """Concatenates two embeddings of format {client_id: embedding} into a new dictionary

        :param embedding1: First embedding dictionary
        :param embedding2: Second embedding dictionary
        :return: Concatenated embedding dictionary
        """
        return {**embedding1, **embedding2}

    def save(self, data_dict, path=None, file_type="npz"):
        if not path:
            path = self.get_tmp_path()

        logger.info("SPLIT LEARNING HELPER: Saving data to {}".format(path))

        # Ensure all values are numpy arrays
        processed_dict = {str(k): np.array(v) for k, v in data_dict.items()}

        with open(path, "wb") as f:
            np.savez_compressed(f, **processed_dict)

        return path

    def load(self, path):
        """Load embeddings/gradients.

        :param path: Path to file
        :return: Dict mapping client IDs to numpy arrays (either embeddings or gradients)
        """
        try:
            data = np.load(path)
            logger.info("SPLIT LEARNING HELPER: loaded data from {}".format(path))
            result_dict = {k: data[k] for k in data.files}
            return result_dict
        except Exception as e:
            logger.error(f"Error in splitlearninghelper: loading data from {path}: {str(e)}")
            raise

    def get_tmp_path(self, suffix=".npz"):
        """Return a temporary output path compatible with save_model, load_model.

        :param suffix: File suffix.
        :return: Path to file.
        """
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        return path

    def check_supported_file_type(self, file_type):
        """Check if the file type is supported.

        :param file_type: File type to check.
        :type file_type: str
        :return: True if supported, False otherwise.
        :rtype: bool
        """
        supported_file_types = ["npz", "raw_binary"]
        if file_type not in supported_file_types:
            raise ValueError("File type not supported. Supported types are: {}".format(supported_file_types))
        return True
