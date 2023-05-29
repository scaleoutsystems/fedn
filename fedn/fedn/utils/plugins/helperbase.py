import os
import tempfile
from abc import ABC, abstractmethod


class HelperBase(ABC):
    """ Abstract class defining helpers. """

    def __init__(self):
        """ Initialize helper. """

        self.name = self.__class__.__name__

    @abstractmethod
    def increment_average(self, model, model_next, a, W):
        """ Compute one increment of incremental weighted averaging.

        :param model: Current model weights in array-like format.
        :param model_next: New model weights in array-like format.
        :param a: Number of examples in new model.
        :param W: Total number of examples.
        :return: Incremental weighted average of model weights.
        """
        pass

    @abstractmethod
    def save(self, model, path):
        """Serialize weights to file. The serialized model must be a single binary object.

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
