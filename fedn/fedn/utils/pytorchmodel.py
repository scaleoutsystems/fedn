import os
import tempfile
import torch
from collections import OrderedDict
from .helpers import HelperBase
from functools import reduce

class PytorchModelHelper(HelperBase):

    def average_weights(self, models):
        weights = models
        avg_w = OrderedDict()
        for name in weights[0].keys():
            tensorList = [weight[name] for weight in weights]
            tensorSum = reduce(torch.add, tensorList)
            avg_w[name] = torch.div(tensorSum, len(weights))
        return avg_w

    def increment_average(self, model, model_next, n):
        """ Update an incremental average. """
        w_0 = self.get_weights(model)
        w_1 = self.get_weights(model_next)
        w = OrderedDict()
        for name in w_0.keys():
            tensorDiff = w_1[name] - w_0[name]
            w[name] = w_0[name] + tensorDiff/n
        self.set_weights(model, w)

    def set_weights(self, model, weights):
        for key, value in weights.items():
            model[key] = value

    def get_weights(self, model):
        return model

    # def get_tmp_path(self):
    #     _ , path = tempfile.mkstemp(suffix='.pth')
    #     return path

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
