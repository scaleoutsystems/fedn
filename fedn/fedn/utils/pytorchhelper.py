import os
import tempfile
from collections import OrderedDict
from io import BytesIO

import numpy as np

from .helpers import HelperBase


class PytorchHelper(HelperBase):

    def increment_average(self, model, model_next, a, W):
        """ Update a weighted incremental average. """
        w = OrderedDict()
        for name in model.keys():
            tensorDiff = model_next[name] - model[name]
            w[name] = model[name] + a*tensorDiff / W
        return w

    def save_model(self, weights_dict, path=None):
        """

        :param weights_dict:
        :param path:
        :return:
        """
        if not path:
            path = self.get_tmp_path()
        np.savez_compressed(path, **weights_dict)
        return path

    def load_model(self, path):
        """

        :param path:
        :return:
        """
        b = np.load(path)
        weights_np = OrderedDict()
        for i in b.files:
            weights_np[i] = b[i]
        return weights_np
