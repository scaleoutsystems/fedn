import os
import tempfile
from io import BytesIO

import numpy as np

from .helpers import HelperBase


class KerasHelper(HelperBase):
    """ FEDn helper class for keras.Sequential. """

    def increment_average(self, weights, weights_next, a, W):
        """ Update a weighted incremental average. """
        w_prev = weights
        w_next = weights_next
        w = np.add(w_prev, a*(np.array(w_next) - np.array(w_prev)) / W)
        return w

    def save_model(self, weights, path=None):
        """

        :param weights:
        :param path:
        :return:
        """
        if not path:
            path = self.get_tmp_path()

        weights_dict = {}
        for i, w in enumerate(weights):
            weights_dict[str(i)] = w

        np.savez_compressed(path, **weights_dict)

        return path

    def load_model(self, path):
        """

        :param path:
        :return:
        """
        a = np.load(path)
        weights = []
        for i in range(len(a.files)):
            weights.append(a[str(i)])
        return weights
