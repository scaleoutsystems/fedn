import os
import tempfile
import torch
from collections import OrderedDict
from .helpers import HelperBase
from functools import reduce

class PytorchModelHelper(HelperBase):

    def increment_average(self, model, model_next, n):
        """ Update an incremental average. """
        w = OrderedDict()
        for name in model.keys():
            tensorDiff = model_next[name] - model[name]
            w[name] = model[name] + tensorDiff/n
        return w


    def get_tmp_path(self):
        _ , path = tempfile.mkstemp(suffix='.pth')
        return path

    def save_model(self, model, path=None):
        if not path:
            path = self.get_tmp_path()
        torch.save(model, path)
        return path

    def load_model(self, path="weights.pth"):
        return torch.load(path)

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
        outfile_name = self.save_model(model)

        from io import BytesIO
        a = BytesIO()
        a.seek(0, 0)
        with open(outfile_name, 'rb') as f:
            a.write(f.read())
        os.unlink(outfile_name)
        return a
