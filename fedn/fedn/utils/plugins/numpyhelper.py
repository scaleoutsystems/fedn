import tempfile

import numpy as np

from .helperbase import HelperBase


class Helper(HelperBase):
    """ FEDn helper class.

    For models that can be serialized as a
    list of numpy ndarrays.

    model has to be on format list of numpy ndarray.

    """

    def __init__(self):
        """ Initialize helper. """
        super().__init__()
        self.name = "numpyhelper"

    def increment_average(self, m1, m2, n, N):
        """ Update an incremental average.

        :param m1: Current model weights.
        :type m1: numpy ndarray.
        :param m2: New model weights.
        :type m2: numpy ndarray.
        :param n: Number of examples in new model.
        :type n: int
        :param N: Total number of examples
        :return: Incremental weighted average of model weights.
        :rtype: :class:`numpy.array`
        """
        return np.add(m1, n*(m2 - m1) / N)

    def add(self, m1, m2, a=1.0, b=1.0):
        """ Add weights.

        :param model: Current model weights with keys from torch state_dict.
        :type model: OrderedDict
        :param model_next: New model weights with keys from torch state_dict.
        :type model_next: OrderedDict
        :return: Incremental weighted average of model weights.
        :rtype: OrderedDict
        """

        w = a*m1 + b*m2
        return w

    def subtract(self, m1, m2, a=1.0, b=1.0):
        """ Subtract model weights m2 from m1.

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

        w = a*m1-b*m2
        return w

    def norm(self, m):
        """ Compute the L2 norm of the weights/model update. """

        return np.linalg.norm(m)

    def save(self, model, path=None):
        """ Serialize weights/parameters to file.

        :param model: Weights/parameters in numpy array format.
        :type model: numpy array.
        :param path: Path to file.
        :type path: str
        :return: Path to file.
        :rtype: str
        """
        if not path:
            _, path = tempfile.mkstemp()
        np.savetxt(path, model)
        return path

    def load(self, path):
        """ Load weights/parameters from file or filelike.

        :param path: Path to file.
        :type path: str
        :return: Weights/parameters in numpy array format.
        :rtype: :class:`numpy.array`
        """
        model = np.loadtxt(path)
        return model
