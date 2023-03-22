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

    def save(self, weights_dict, path=None):
        """ Serialize weights to file.

        :param weights_dict:
        :param path: File path.
        :return: Path to file (generated as tmp file unless path is set).
        """
        if not path:
            path = self.get_tmp_path()
        np.savez_compressed(path, **weights_dict)
        return path

    def load(self, fh):
        """ Load weights from file or filelike.

        :param fh: file path, filehandle, filelike.
        :return: OrderedDict containing weights in numpy format.
        """
        a = np.load(fh)
        weights_np = OrderedDict()
        for i in a.files:
            weights_np[i] = a[i]
        return weights_np
