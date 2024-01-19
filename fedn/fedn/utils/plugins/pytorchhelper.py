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

    def add(self, m1, m2, a=1.0, b=1.0):
        """ m1*a + m2*b

        :param model: Current model weights with keys from torch state_dict.
        :type model: OrderedDict
        :param model_next: New model weights with keys from torch state_dict.
        :type model_next: OrderedDict
        :return: Incremental weighted average of model weights.
        :rtype: OrderedDict
        """
        w = OrderedDict()
        for name in m1.keys():
            tensorSum = a*m1[name] + b*m2[name]
            w[name] = tensorSum
        return w

    def subtract(self, m1, m2, a=1.0, b=1.0):
        """ m1*a - m2*b.

        :param m1: Current model weights with keys from torch state_dict.
        :type m1: OrderedDict
        :param m2: New model weights with keys from torch state_dict.
        :type m2: OrderedDict
        :return: m1*a-m2*b
        :rtype: OrderedDict
        """
        return self.add(m1, m2, a, -b)

    def divide(self, m1, m2):
        """ Subtract weights.

        :param m1: Current model weights with keys from torch state_dict.
        :type m1: OrderedDict
        :param m2: New model weights with keys from torch state_dict.
        :type m2: OrderedDict
        :return: m1/m2.
        :rtype: OrderedDict
        """

        res = OrderedDict()
        for key, val in m1.items():
            res[key] = np.divide(val, m2[key])

        return res

    def multiply(self, m1, m2):
        """ Multiply m1 by m2.

        :param m1: Current model weights with keys from torch state_dict.
        :type m1: OrderedDict
        :param m2: New model weights with keys from torch state_dict.
        :type m2: OrderedDict
        :return: m1.*m2 
        :rtype: OrderedDict
        """

        res = OrderedDict()
        for key, val in m1.items():
            res[key] = np.multiply(np.array(val), m2)

        return res

    def sqrt(self, m1):
        """ Sqrt of m1, element-wise.

        :param m1: Current model weights with keys from torch state_dict.
        :type model: OrderedDict
        :param model_next: New model weights with keys from torch state_dict.
        :type model_next: OrderedDict
        :return: sqrt(m1)
        :rtype: OrderedDict
        """
        res = OrderedDict()
        for key, val in m1.items():
            res[key] = np.sqrt(np.array(val))

        return res

    def power(self, m1, a):
        """ m1 raised to the power of m2.

        :param m1: Current model weights with keys from torch state_dict.
        :type m1: OrderedDict
        :param m2: New model weights with keys from torch state_dict.
        :type a: float
        :return: m1.^m2
        :rtype: OrderedDict
        """
        res = OrderedDict()
        for key, val in m1.items():
            res[key] = np.power(val, a)

        return res

    def norm(self, m):
        """Compute the L1-norm of the tensor m. """
        n = 0.0
        for name, val in m.items():
            n += np.linalg.norm(np.array(val), 1)

        return n

    def ones(self, m1, a):
        res = OrderedDict()
        for key, val in m1.items():
            res[key] = np.ones(np.shape(val))*a

        return res

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
        :rtype: OrderedDict
        """
        a = np.load(path)
        weights_np = OrderedDict()
        for i in a.files:
            weights_np[i] = a[i]
        return weights_np
