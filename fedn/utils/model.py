import tempfile
import threading

from fedn.utils.checksum import compute_checksum_from_stream
from fedn.utils.helpers.plugins.numpyhelper import Helper

CHUNK_SIZE = 8192  # 8 KB chunk size for reading/writing files


class FednModel:
    """The FednModel class is the primary model representation in the FEDn framework.
    A FednModel object contains a data object (tempfile.SpooledTemporaryFile) that holds the model parameters.
    The model parameters dict can be extracted from the data object or be used to create a model object.
    Unpacking of the model parameters is done by the helper which needs to be provided either to the the class or
    to the method
    """

    def __init__(self):
        """Initializes a FednModel object."""
        # Using SpooledTemporaryFile to handle large model data efficiently
        # It will automatically store on disk if the data exceeds the specified size (10 MB in this case)
        self._data = tempfile.SpooledTemporaryFile(10 * 1024 * 1024)  # 10 MB temporary file
        self._data_lock = threading.RLock()
        self.model_id = None
        self.helper = None
        self._checksum = None

    @property
    def checksum(self) -> str:
        """Returns the checksum of the model data."""
        if self._checksum is None:
            self._checksum = compute_checksum_from_stream(self.get_stream())
        return self._checksum

    def verify_checksum(self, checksum: str) -> bool:
        """Verifies the checksum of the model data.

        If no checksum is provided, it returns True.
        """
        return checksum is None or self.checksum == checksum

    def get_stream(self):
        """Returns a stream of the model data.

        To avoid concurrency issues, a new stream is created each time this method is called.
        """
        with self._data_lock:
            self._data.seek(0)
            new_stream = tempfile.SpooledTemporaryFile(max_size=self._data._max_size)
            while chunk := self._data.read(CHUNK_SIZE):
                new_stream.write(chunk)
            new_stream.seek(0)
            self._data.seek(0)
        return new_stream

    def get_stream_unsafe(self):
        """Returns the internal stream of the model data.

        This method is not thread-safe and should be used with caution.
        """
        with self._data_lock:
            self._data.seek(0)
            return self._data

    def get_model_params(self, helper=None):
        """Returns the model parameters as a dictionary."""
        stream = self.get_stream()
        self.helper = helper or self.helper
        if self.helper is None:
            raise ValueError("No helper provided to unpack model parameters.")
        return self.helper.load(stream)

    def save_to_file(self, file_path: str):
        """Saves the model data to a file."""
        with open(file_path, "wb") as file:
            stream = self.get_stream()
            while chunk := stream.read(CHUNK_SIZE):
                file.write(chunk)

    @staticmethod
    def from_model_params(model_params: dict, helper=None) -> "FednModel":
        """Creates a FednModel from model parameters."""
        model_reference = FednModel()
        model_reference.helper = helper
        if helper is None:
            # No helper provided, using numpy helper as default
            helper = Helper()
        helper.save(model_params, model_reference._data)
        model_reference._data.seek(0)
        return model_reference

    @staticmethod
    def from_file(file_path: str) -> "FednModel":
        """Creates a FednModel from a file."""
        with open(file_path, "rb") as file:
            return FednModel.from_stream(file)

    @staticmethod
    def from_stream(stream) -> "FednModel":
        """Creates a FednModel from a stream."""
        model_reference = FednModel()
        while chunk := stream.read(CHUNK_SIZE):
            model_reference._data.write(chunk)
        model_reference._data.seek(0)
        return model_reference

    @staticmethod
    def from_chunk_generator(chunk_generator) -> "FednModel":
        """Creates a FednModel from a chunk generator."""
        model_reference = FednModel()
        for chunk in chunk_generator:
            model_reference._data.write(chunk)
        model_reference._data.seek(0)
        return model_reference
