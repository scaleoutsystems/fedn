import os
import tempfile
import numpy as np
import tensorflow.keras.models as krm
import collections
import tempfile

from .helpers import HelperBase

class KerasWeightsHelper(HelperBase):
    """ FEDn helper class for keras.Sequential. """

    def average_weights(self, outer_models):
        """ Average weights of Keras Sequential models. """
        weights = [model['model'].get_weights() for model in outer_models]

        avg_w = []
        for l in range(len(weights[0])):
            lay_l = np.array([w[l] for w in weights])
            weight_l_avg = np.mean(lay_l, 0)
            avg_w.append(weight_l_avg)

        return avg_w

    def increment_average(self, model, model_next, n):
        """ Update an incremental average. """
        w_prev = self.get_weights(model)
        w_next = self.get_weights(model_next)
        w = np.add(w_prev, (np.array(w_next) - np.array(w_prev)) / n)
        self.set_weights(model, w)

    def set_weights(self, model, weights):
        model['model'].set_weights(weights)

    def get_weights(self, model):
        return model['model'].get_weights()

    def get_tmp_path(self):
        fod, path = tempfile.mkstemp(suffix='.h5')
        return path

    def get_model_struct(self):
        fod, path = tempfile.mkstemp(prefix='kerasmodel')

    def save_model(self, outer_model, path=None):
        import tarfile
        model = outer_model['model']

        if not path:
            _ , path = tempfile.mkstemp()

        _, package_path = tempfile.mkstemp()
        with open(package_path, 'wb') as fh:
            fh.write(outer_model['package'].getbuffer())
            fh.flush()
        tar = tarfile.open(package_path, "r:gz")
        tempf = tempfile.TemporaryDirectory()
        tar.extractall(tempf.name)

        _, weights_path = tempfile.mkstemp(suffix='.h5')
        # saving model weights
        model.save_weights(weights_path)

        tar = tarfile.open(path, "w:gz")
        tar.add(os.path.join(tempf.name,'src'),'src')

        tar.add(weights_path,'weights.h5')
        tar.close()
        return path

    def load_model(self, path="model"):
        import importlib.util
        import tarfile
        import sys


        # unzip
        tempf = tempfile.TemporaryDirectory()
        tar = tarfile.open(path, "r:gz")
        tar.extractall(tempf.name)
        tar.close()

        sys.path.append(os.path.join(tempf.name,'src'))

        from kerasmodel import create_seed_model

        model = create_seed_model(os.path.join(tempf.name,'src'))
        model.load_weights(os.path.join(tempf.name, 'weights.h5'))

        outer_model = {}
        outer_model['model'] = model

        # zip the src folder
        fod, temp_path = tempfile.mkstemp()
        tar = tarfile.open(temp_path, "w:gz")
        tar.add(os.path.join(tempf.name, 'src'), 'src')
        tar.close()

        #create a bytesio from the zip directory
        from io import BytesIO
        a = BytesIO()
        a.seek(0, 0)
        with open(temp_path, 'rb') as f:
            a.write(f.read())
        outer_model['package'] = a
        outer_model['path'] = tempf.name
        return outer_model

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