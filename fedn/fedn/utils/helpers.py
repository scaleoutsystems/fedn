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


def get_helper(helper_type):
    """ Return an instance of the helper class.

    :param helper_type (str): The helper type ('keras','pytorch')
    :return:
    """
    if helper_type == 'numpyarray':
        # TODO: refactor cyclical import to avoid this ugly line
        """ noqa """; from fedn.utils.numpyarrayhelper import NumpyArrayHelper  # autopep8: off # noqa: E702
        return NumpyArrayHelper()
    elif helper_type == 'keras':
        """ noqa """; from fedn.utils.kerashelper import KerasHelper  # autopep8: off # noqa: E702
        return KerasHelper()
    elif helper_type == 'pytorch':
        """ noqa """; from fedn.utils.pytorchhelper import PytorchHelper  # autopep8: off # noqa: E702
        return PytorchHelper()
    else:
        return None
