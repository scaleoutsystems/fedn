import os
import tempfile
from io import BytesIO

import numpy as np

from .helpers import HelperBase


class KerasHelper(HelperBase):
    """ FEDn helper class for keras.Sequential. """

    def average_weights(self, weights):
        """ Average weights of Keras Sequential models. """

        avg_w = []
        for i in range(len(weights[0])):
            lay_l = np.array([w[i] for w in weights])
            weight_l_avg = np.mean(lay_l, 0)
            avg_w.append(weight_l_avg)

        return avg_w

    def increment_average(self, weights, weights_next, n):
        """ Update an incremental average. """
        w_prev = weights
        w_next = weights_next
        w = np.add(w_prev, (np.array(w_next) - np.array(w_prev)) / n)
        return w

    def set_weights(self, weights_, weights):
        """

        :param weights_:
        :param weights:
        """
        weights_ = weights  # noqa F841

    def get_weights(self, weights):
        """

        :param weights:
        :return:
        """
        return weights

    def get_tmp_path(self):
        """ Return a temporary output path compatible with save_model, load_model. """
        fd, path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)
        return path

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

    def load_model(self, path="weights.npz"):
        """

        :param path:
        :return:
        """
        a = np.load(path)
        weights = []
        for i in range(len(a.files)):
            weights.append(a[str(i)])
        return weights

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
