import os
import tempfile
from collections import OrderedDict
from io import BytesIO

import numpy as np

from .helpers import HelperBase


class PytorchHelper(HelperBase):

    def increment_average(self, model, model_next, n):
        """ Update an incremental average. """
        w = OrderedDict()
        for name in model.keys():
            tensorDiff = model_next[name] - model[name]
            w[name] = model[name] + tensorDiff / n
        return w

    def get_tmp_path(self):
        """

        :return:
        """
        fd, path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)
        return path

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

    def load_model(self, path="weights.npz"):
        """

        :param path:
        :return:
        """
        b = np.load(path)
        weights_np = OrderedDict()
        for i in b.files:
            weights_np[i] = b[i]
        return weights_np

    def load_model_from_BytesIO(self, model_bytesio):
        """ Load a model from a BytesIO object. """
        path = self.get_tmp_path()
        with open(path, 'wb') as fh:
            fh.write(model_bytesio)
            fh.flush()
        model = self.load_model(path)
        os.unlink(path)
        return model

    def serialize_model_to_BytesIO(self, model):
        """

        :param model:
        :return:
        """
        outfile_name = self.save_model(model)

        a = BytesIO()
        a.seek(0, 0)
        with open(outfile_name, 'rb') as f:
            a.write(f.read())
        os.unlink(outfile_name)
        return a
