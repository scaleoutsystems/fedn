import collections
from abc import ABC, abstractmethod
import os 
import tempfile


class HelperBase(ABC):
    """ Abstract class defining helpers. """

    def __init__(self):
        """ """

    @abstractmethod
    def increment_average(self, model, model_next, n):
        """ Compute one increment of incremental averaging. 
            n: the iteration index 1...N  in the sequence. 
        """
        pass

    @abstractmethod
    def save_model(self, model, path):
        """
            Serialize the model to file on disk on path.
            The serialized model must be a single binary object.
        """
        pass

    @abstractmethod
    def load_model(self, path):
        """ Load the model save with save_model from disk on path.  """
        pass

    @abstractmethod
    def serialize_model_to_BytesIO(self, model):
        """ Serialize a model to a BytesIO buffered object. """
        pass

    @abstractmethod
    def load_model_from_BytesIO(self, model_bytesio):
        """ Load a model from a BytesIO buffered object. """
        pass

    @abstractmethod
    def get_tmp_path(self):
        """ Return a temporary output path compatible with save_model, load_model. """
        pass

def get_helper(helper_type):
    if helper_type == 'numpyarray':
        from fedn.utils.numpyarrayhelper import NumpyArrayHelper
        return NumpyArrayHelper()
    elif helper_type == 'keras':
        from fedn.utils.kerashelper import KerasHelper
        return KerasHelper()
    elif helper_type == 'pytorch':
        from fedn.utils.pytorchhelper import PytorchHelper
        return PytorchHelper()
    else:
        return None
