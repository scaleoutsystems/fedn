
import numpy as np

from fedn.utils.helpers.helperbase import HelperBase


class Helper(HelperBase):
    """ FEDn helper class for pytorch models. """

    def __init__(self):
        """ Initialize helper. """
        super().__init__()
        self.name = "pytorchhelper"

    def increment_average(self, m1, m2, n, N):
        """ Update a weighted incremental average of model weights.

        :param m1: Current parameters.
        :type model: list of numpy ndarray
        :param m2: next parameters.
        :type model_next: list of numpy ndarray
        :param n: Number of examples used for updating m2.
        :type n: int
        :param N: Total number of examples (accumulated).
        :type N: int
        :return: Updated incremental weighted average.
        :rtype: list of numpy ndarray
        """

        return [np.add(x, n*(y-x)/N) for x, y in zip(m1, m2)]

    def add(self, m1, m2, a=1.0, b=1.0):
        """ m1*a + m2*b

        :param model: Current model weights with keys from torch state_dict.
        :type model: OrderedDict
        :param model_next: New model weights with keys from torch state_dict.
        :type model_next: OrderedDict
        :return: Incremental weighted average of model weights.
        :rtype: OrderedDict
        """

        return [x*a+y*b for x, y in zip(m1, m2)]

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

        return [np.divide(x, y) for x, y in zip(m1, m2)]

    def multiply(self, m1, m2):
        """ Multiply m1 by m2.

        :param m1: Current model weights with keys from torch state_dict.
        :type m1: OrderedDict
        :param m2: New model weights with keys from torch state_dict.
        :type m2: OrderedDict
        :return: m1.*m2
        :rtype: OrderedDict
        """

        return [np.multiply(x, y) for (x, y) in zip(m1, m2)]

    def sqrt(self, m1):
        """ Sqrt of m1, element-wise.

        :param m1: Current model weights with keys from torch state_dict.
        :type model: OrderedDict
        :param model_next: New model weights with keys from torch state_dict.
        :type model_next: OrderedDict
        :return: sqrt(m1)
        :rtype: OrderedDict
        """

        return [np.sqrt(x) for x in m1]

    def power(self, m1, a):
        """ m1 raised to the power of m2.

        :param m1: Current model weights with keys from torch state_dict.
        :type m1: OrderedDict
        :param m2: New model weights with keys from torch state_dict.
        :type a: float
        :return: m1.^m2
        :rtype: OrderedDict
        """

        return [np.power(x, a) for x in m1]

    def norm(self, m):
        """Compute the L1 norm of m. """
        n = 0.0
        for x in m:
            n += np.linalg.norm(x, 1)
        return n

    def ones(self, m1, a):

        res = []
        for x in m1:
            res.append(np.ones(np.shape(x))*a)
        return res

    def save(self, weights, path=None):
        """ Serialize weights to file. The serialized model must be a single binary object.

        :param weights: List of weights in numpy format.
        :param path: Path to file.
        :return: Path to file.
        """
        if not path:
            path = self.get_tmp_path()

        weights_dict = {}
        for i, w in enumerate(weights):
            weights_dict[str(i)] = w

        np.savez_compressed(path, **weights_dict)

        return path

    def load(self, fh):
        """ Load weights from file or filelike.

        :param fh: file path, filehandle, filelike.
        :return: List of weights in numpy format.
        """
        a = np.load(fh)

        weights = []
        for i in range(len(a.files)):
            weights.append(a[str(i)])
        return weights
