import os
import tempfile
from abc import ABC, abstractmethod


class HelperBase(ABC):
    """ Abstract class defining helpers. """

    def __init__(self):
        """ Initialize helper. """

        self.name = self.__class__.__name__

    @abstractmethod
    def increment_average(self, m1, m2, a, W):
        """ Compute one increment of incremental weighted averaging.

        :param m1: Current model weights in array-like format.
        :param m2: New model weights in array-like format.
        :param a: Number of examples in new model.
        :param W: Total number of examples.
        :return: Incremental weighted average of model weights.
        """
        pass

    @abstractmethod
    def add(self, m1, m2, a=1.0, b=1.0):
        """ m1*a + m2*b

        :param model: Current model weights with keys from torch state_dict.
        :type model: OrderedDict
        :param model_next: New model weights with keys from torch state_dict.
        :type model_next: OrderedDict
        :return: Incremental weighted average of model weights.
        :rtype: OrderedDict
        """
        pass

    @abstractmethod
    def subtract(self, m1, m2, a=1.0, b=1.0):
        """ m1*a - m2*b.

        :param m1: Current model weights with keys from torch state_dict.
        :type m1: OrderedDict
        :param m2: New model weights with keys from torch state_dict.
        :type m2: OrderedDict
        :return: m1*a-m2*b
        :rtype: OrderedDict
        """
        pass

    @abstractmethod
    def divide(self, m1, m2):
        """ Subtract weights.

        :param m1: Current model weights with keys from torch state_dict.
        :type m1: OrderedDict
        :param m2: New model weights with keys from torch state_dict.
        :type m2: OrderedDict
        :return: m1/m2.
        :rtype: OrderedDict
        """
        pass

    @abstractmethod
    def multiply(self, m1, m2):
        """ Multiply m1 by m2.

        :param m1: Current model weights with keys from torch state_dict.
        :type m1: OrderedDict
        :param m2: New model weights with keys from torch state_dict.
        :type m2: OrderedDict
        :return: m1.*m2
        :rtype: OrderedDict
        """

        pass

    @abstractmethod
    def sqrt(self, m1):
        """ Sqrt of m1, element-wise.

        :param m1: Current model weights with keys from torch state_dict.
        :type model: OrderedDict
        :param model_next: New model weights with keys from torch state_dict.
        :type model_next: OrderedDict
        :return: sqrt(m1)
        :rtype: OrderedDict
        """
        pass

    @abstractmethod
    def power(self, m1, a):
        """ m1 raised to the power of m2.

        :param m1: Current model weights with keys from torch state_dict.
        :type m1: OrderedDict
        :param m2: New model weights with keys from torch state_dict.
        :type a: float
        :return: m1.^m2
        :rtype: OrderedDict
        """
        pass

    @abstractmethod
    def norm(self, m):
        """Compute the L1-norm of the tensor m. """
        pass

    @abstractmethod
    def ones(self, m1, a):
        """ Return ones times a with same shape as m1. """
        pass

    @abstractmethod
    def save(self, model, path):
        """ Serialize weights to file. The serialized model must be a single binary object.

        :param model: Weights in array-like format.
        :param path: Path to file.

        """
        pass

    @abstractmethod
    def load(self, fh):
        """ Load weights from file or filelike.

        :param fh: file path, filehandle, filelike.
        :return: Weights in array-like format.
        """
        pass

    def get_tmp_path(self):
        """ Return a temporary output path compatible with save_model, load_model.

        :return: Path to file.
        """
        fd, path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)
        return path
