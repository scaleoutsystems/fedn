import os
import tempfile
from io import BytesIO

import numpy as np

from .helpers import HelperBase


class NumpyArrayHelper(HelperBase):
    """ FEDn helper class for numpy arrays. """

    def increment_average(self, model, model_next, n):
        """ Update an incremental average. """
        return np.add(model, (model_next - model) / n)

    def save_model(self, model, path=None):
        """

        :param model:
        :param path:
        :return:
        """
        if not path:
            _, path = tempfile.mkstemp()
        np.savetxt(path, model)
        return path

    def load_model(self, path):
        """

        :param path:
        :return:
        """
        model = np.loadtxt(path)
        return model
