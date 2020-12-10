import os
import tempfile
import numpy as np
import tensorflow.keras.models as krm
import collections
import tempfile

from .helpers import HelperBase

class KerasWeightsHelper(HelperBase):
    """ FEDn helper class for keras.Sequential. """

    def average_weights(self, weights):
        """ Average weights of Keras Sequential models. """
        #weights = [w for w in weights]

        avg_w = []
        for l in range(len(weights[0])):
            lay_l = np.array([w[l] for w in weights])
            weight_l_avg = np.mean(lay_l, 0)
            avg_w.append(weight_l_avg)

        return avg_w

    def increment_average(self, weights, weights_next, n):
        """ Update an incremental average. """
        w_prev = weights
        w_next = weights_next
        w = np.add(w_prev, (np.array(w_next) - np.array(w_prev)) / n)
        weights = w

    def set_weights(self, weights_, weights):
        weights_ = weights

    def get_weights(self, weights):
        return weights

    def get_tmp_path(self):
        fod, path = tempfile.mkstemp(suffix='.npz')
        return path

    def get_model_struct(self):
        fod, path = tempfile.mkstemp(prefix='kerasmodel')

    def save_model(self, weights, path=None):
        import tarfile
        #model = outer_model['model']

        if not path:
            path = self.get_tmp_path()
        np.savez_compressed(path, weights=weights)

        return path

    def load_model(self, path="weights.npz"):

        weights = np.load(path, allow_pickle=True)['weights'].tolist()
        return weights

    def load_model_from_BytesIO(self, model_bytesio):
        """ Load a model from a BytesIO object. """
        path = self.get_tmp_path()
        with open(path, 'wb') as fh:
            fh.write(model_bytesio)
            fh.flush()

        return self.load_model(path)

    def serialize_model_to_BytesIO(self, model):
        outfile_name = self.save_model(model)

        from io import BytesIO
        a = BytesIO()
        a.seek(0, 0)
        with open(outfile_name, 'rb') as f:
            a.write(f.read())
        os.unlink(outfile_name)
        return a