import tempfile

import numpy as np

from .helperbase import HelperBase


class Helper(HelperBase):
    """ FEDn helper class for numpy arrays. """

    def increment_average(self, model, model_next, n):
        """ Update an incremental average. 

        :param model: Current model weights.
        :type model: numpy array.
        :param model_next: New model weights.
        :type model_next: numpy array.
        :param n: Number of examples in new model.
        :type n: int
        :return: Incremental weighted average of model weights.
        :rtype: :class:`numpy.array`
        """
        return np.add(model, (model_next - model) / n)

    def save(self, model, path=None):
        """Serialize weights/parameters to file.

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
        """Load weights/parameters from file or filelike.

        :param path: Path to file.
        :type path: str
        :return: Weights/parameters in numpy array format.
        :rtype: :class:`numpy.array`
        """
        model = np.loadtxt(path)
        return model
