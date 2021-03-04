import os
import tempfile
import numpy as np
import pickle
import collections
import tempfile

from .helpers import HelperBase

class NumpyArrayHelper(HelperBase):
    """ FEDn helper class for numpy arrays. """

    def increment_average(self, model, model_next, n):
        """ Update an incremental average. """
        return np.add(model, (model_next - model) / n)

    def save_model(self, model, path=None):
        if not path:
            _ , path = tempfile.mkstemp()
        np.savetxt(path,model)
        return path

    def load_model(self, path):
        model = np.loadtxt(path)
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

    def get_tmp_path(self):
        """ Return a temporary output path compatible with save_model, load_model. """
        fd, path = tempfile.mkstemp()
        os.close(fd)
        return path

    def load_model_from_BytesIO(self,model_bytesio):
        """ Load a model from a BytesIO object. """
        path = self.get_tmp_path()
        with open(path, 'wb') as fh:
            fh.write(model_bytesio)
            fh.flush()
        model = np.loadtxt(path)
        os.unlink(path)
        return model
