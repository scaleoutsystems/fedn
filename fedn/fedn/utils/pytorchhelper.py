import os
import tempfile
from collections import OrderedDict
from .helpers import HelperBase
from functools import reduce
import numpy as np


class PytorchHelper(HelperBase):

    def increment_average(self, model_pack_a, model_pack_b, n):
        """ Update an incremental average. """
        print("---NEW INCREMENT AVERAGE-------")
        weights_a = model_pack_a['weights']
        k_a = model_pack_a['k']
        weights_b = model_pack_b['weights']
        k_b = model_pack_b['k']

        weights_c = OrderedDict()
        for name in weights_a.keys():
            weights_c[name] = k_a / (k_a + k_b) * weights_a[name] + k_b / (k_a + k_b) * weights_b[name]
        k_c = k_a + k_b
        pack_c = {'weights': weights_c, 'k': k_c}
        return pack_c

    def get_tmp_path(self):
        fd , path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)
        return path

    def save_model(self, model_pack, path=None):
        print("---NEW SAVE MODEL-------")

        if not path:
            path = self.get_tmp_path()
        weights_dict = model_pack['weights']
        weights_dict['weight_factor'] = model_pack['k']
        np.savez_compressed(path, **weights_dict)
        return path

    def load_model(self, path="weights.npz"):
        print("---NEW LOAD MODEL-------")

        b = np.load(path)
        weights_np = OrderedDict()
        for i in b.files:
            if i == 'weight_factor':
                weight_factor = i
            else:
                weights_np[i] = b[i]

        model_pack = {'weights': weights_np, 'k': weight_factor}
        return model_pack

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
