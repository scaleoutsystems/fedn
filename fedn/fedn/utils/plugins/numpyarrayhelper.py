import tempfile

import numpy as np

from .helperbase import HelperBase


class Helper(HelperBase):
    """ FEDn helper class for numpy arrays. """

    def increment_average(self, model, model_next, n):
        """ Update an incremental average. """
        return np.add(model, (model_next - model) / n)

    def save(self, model, path=None):
        """Serialize weights/parameters to file.

        :param model:
        :param path:
        :return:
        """
        if not path:
            _, path = tempfile.mkstemp()
        np.savetxt(path, model)
        return path

    def load(self, path):
        """Load weights/parameters from file or filelike.

        :param path:
        :return:
        """
        model = np.loadtxt(path)
        return model
