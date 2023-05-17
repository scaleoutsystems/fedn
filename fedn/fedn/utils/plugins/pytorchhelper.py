from collections import OrderedDict

import numpy as np

from .helperbase import HelperBase


class Helper(HelperBase):
    """ FEDn helper class for pytorch. """

    def __init__(self):
        """ Initialize helper. """
        super().__init__()
        self.name = "pytorchhelper"

    def increment_average(self, model, model_next, num_examples, total_examples):
        """ Update a weighted incremental average of model weights.

        :param model: Current model weights with keys from torch state_dict.
        :type model: OrderedDict
        :param model_next: New model weights with keys from torch state_dict.
        :type model_next: OrderedDict
        :param num_examples: Number of examples in new model.
        :type num_examples: int
        :param total_examples: Total number of examples.
        :type total_examples: int
        :return: Incremental weighted average of model weights.
        :rtype: OrderedDict
        """
        w = OrderedDict()
        for name in model.keys():
            tensorDiff = model_next[name] - model[name]
            w[name] = model[name] + num_examples*tensorDiff / total_examples
        return w

    def save(self, model, path=None):
        """ Serialize weights to file. The serialized model must be a single binary object.

        :param model: Weights of model with keys from torch state_dict.
        :type model: OrderedDict
        :param path: File path.
        :type path: str
        :return: Path to file (generated as tmp file unless path is set).
        :rtype: str
        """
        if not path:
            path = self.get_tmp_path()
        np.savez_compressed(path, **model)
        return path

    def load(self, path):
        """ Load weights from file or filelike.

        :param path: file path, filehandle, filelike.
        :type path: str
        :return: Weights of model with keys from torch state_dict.
        """
        a = np.load(path)
        weights_np = OrderedDict()
        for i in a.files:
            weights_np[i] = a[i]
        return weights_np
