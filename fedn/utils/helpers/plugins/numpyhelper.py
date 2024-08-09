import os
import tempfile
from io import BytesIO

import numpy as np

from fedn.utils.helpers.helperbase import HelperBase


class Helper(HelperBase):
    """FEDn helper class for models weights/parameters that can be transformed to numpy ndarrays."""

    def __init__(self):
        """Initialize helper."""
        super().__init__()
        self.name = "numpyhelper"

    def increment_average(self, m1, m2, n, N):
        """Update a weighted incremental average of model weights.

        :param m1: Current parameters.
        :type model: list of numpy ndarray
        :param m2: next parameters.
        :type model_next: list of numpy ndarray
        :param n: Number of examples used for updating m2.
        :type n: int
        :param N: Total number of examples (accumulated).
        :type N: int
        :return: Updated incremental weighted average.
        :rtype: list of numpy ndarray
        """
        return [np.add(x, n * (y - x) / N) for x, y in zip(m1, m2)]

    def add(self, m1, m2, a=1.0, b=1.0):
        """m1*a + m2*b

        :param model: Current model weights.
        :type model: list of ndarrays
        :param model_next: New model weights.
        :type model_next: list of ndarrays
        :return: Incremental weighted average of model weights.
        :rtype: list of ndarrays
        """
        return [x * a + y * b for x, y in zip(m1, m2)]

    def subtract(self, m1, m2, a=1.0, b=1.0):
        """m1*a - m2*b.

        :param m1: Current model weights.
        :type m1: list of ndarrays
        :param m2: New model weights.
        :type m2: list of ndarrays
        :return: m1*a-m2*b
        :rtype: list of ndarrays
        """
        return self.add(m1, m2, a, -b)

    def divide(self, m1, m2):
        """Subtract weights.

        :param m1: Current model weights.
        :type m1: list of ndarrays
        :param m2: New model weights.
        :type m2: list of ndarrays
        :return: m1/m2.
        :rtype: list of ndarrays
        """
        return [np.divide(x, y) for x, y in zip(m1, m2)]

    def multiply(self, m1, m2):
        """Multiply m1 by m2.

        :param m1: Current model weights.
        :type m1: list of ndarrays
        :param m2: New model weights.
        :type m2: list of ndarrays
        :return: m1.*m2
        :rtype: list of ndarrays
        """
        return [np.multiply(x, y) for (x, y) in zip(m1, m2)]

    def sqrt(self, m1):
        """Sqrt of m1, element-wise.

        :param m1: Current model weights.
        :type model: list of ndarrays
        :param model_next: New model weights.
        :type model_next: list of ndarrays
        :return: sqrt(m1)
        :rtype: list of ndarrays
        """
        return [np.sqrt(x) for x in m1]

    def power(self, m1, a):
        """m1 raised to the power of m2.

        :param m1: Current model weights.
        :type m1: list of ndarrays
        :param m2: New model weights.
        :type a: float
        :return: m1.^m2
        :rtype: list of ndarrays
        """
        return [np.power(x, a) for x in m1]

    def norm(self, m):
        """Return the norm (L1) of model weights.

        :param m: Current model weights.
        :type m: list of ndarrays
        :return: norm of m
        :rtype: float
        """
        n = 0.0
        for x in m:
            n += np.linalg.norm(x, 1)
        return n

    def sign(self, m):
        """Sign of m.

        :param m: Model parameters.
        :type m: list of ndarrays
        :return: sign(m)
        :rtype: list of ndarrays
        """
        return [np.sign(x) for x in m]

    def ones(self, m1, a):
        """Return a list of numpy arrays of the same shape as m1, filled with ones.

        :param m1: Current model weights.
        :type m1: list of ndarrays
        :param a: Scalar value.
        :type a: float
        :return: list of numpy arrays of the same shape as m1, filled with ones.
        :rtype: list of ndarrays
        """
        res = []
        for x in m1:
            res.append(np.ones(np.shape(x)) * a)
        return res

    def save(self, weights, path=None, file_type="npz"):
        """Serialize weights to file. The serialized model must be a single binary object.

        :param weights: List of weights in numpy format.
        :param path: Path to file.
        :param file_type: File type to save to. Can be 'npz' or 'raw_binary'. Default is 'npz'.
        :return: Path to file.
        """
        self.check_supported_file_type(file_type)

        if file_type == "npz":
            if not path:
                path = self.get_tmp_path()

            weights_dict = {}
            for i, w in enumerate(weights):
                weights_dict[str(i)] = w

            np.savez_compressed(path, **weights_dict)
            return path
        else:
            if not path:
                path = self.get_tmp_path(suffix=".bin")
            weights = np.concatenate(weights)
            weights.tofile(path)
            return path

    def load(self, path, file_type="npz"):
        """Load weights from file or filelike.

        :param path: file path, filehandle, filelike.
        :return: List of weights in numpy format.
        """
        self.check_supported_file_type(file_type)
        weights = []
        if file_type == "npz":
            a = np.load(path)
            for i in range(len(a.files)):
                weights.append(a[str(i)])
        else:
            if isinstance(path, BytesIO):
                a = np.frombuffer(path.read(), dtype=np.float64)
            else:
                a = np.fromfile(path, dtype=np.float64)
            weights.append(a)
        return weights

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
