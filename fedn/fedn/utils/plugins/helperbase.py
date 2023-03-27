import os
import tempfile
from abc import ABC, abstractmethod
from io import BytesIO


class HelperBase(ABC):
    """ Abstract class defining helpers. """

    def __init__(self):
        """ """

    @abstractmethod
    def increment_average(self, model, model_next, a, W):
        """ Compute one increment of incremental weighted averaging.
        """
        pass

    @abstractmethod
    def save(self, model, path):
        """
            Serialize weights to file.
            The serialized model must be a single binary object.
        """
        pass

    @abstractmethod
    def load(self, fh):
        """ Load weights from file or filelike.  """
        pass

    def get_tmp_path(self):
        """ Return a temporary output path compatible with save_model, load_model. """
        fd, path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)
        return path
