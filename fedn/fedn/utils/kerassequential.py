import os
import tempfile
import numpy as np
import tensorflow.keras.models as krm
import collections
import tempfile

from .helpers import HelperBase

class KerasSequentialHelper(HelperBase):
    """ FEDn helper class for keras.Sequential. """

    def increment_average(self, model, model_next, n):
        """ Update an incremental average. """
        w_prev = self.get_weights(model)
        w_next = self.get_weights(model_next)
        w = np.add(w_prev, (np.array(w_next) - np.array(w_prev)) / n)
        self.set_weights(model, w)
        return model

    def set_weights(self, model, weights):
        model.set_weights(weights)

    def get_weights(self, model):
        return model.get_weights()

    def get_tmp_path(self):
        """ Return a temporary output path compatible with save_model, load_model. """
        fd, path = tempfile.mkstemp(suffix='.h5')
        os.close(fd)
        return path

    def save_model(self, model, path=None):
        if not path:
            path = self.get_tmp_path()
            model.save(path)
        else:
            model.save(path)

        return path

    def load_model(self, path):
        model = krm.load_model(path)
        return model

    def serialize_model_to_BytesIO(self,model):

        outfile_name = self.save_model(model)

        from io import BytesIO
        a = BytesIO()
        a.seek(0, 0)
        with open(outfile_name, 'rb') as f:
            a.write(f.read())
        os.unlink(outfile_name)
        return a

    def load_model_from_BytesIO(self,model_bytesio):
        """ Load a model from a BytesIO object. """
        path = self.get_tmp_path()
        with open(path, 'wb') as fh:
            fh.write(model_bytesio)
            fh.flush()

        model = self.load_model(path)
        os.unlink(path)
        return model
