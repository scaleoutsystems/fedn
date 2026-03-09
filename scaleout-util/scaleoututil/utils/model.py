import tempfile
import threading
from typing import BinaryIO, Iterable

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
from scaleoututil.utils.checksum import compute_checksum_from_stream
from scaleoututil.helpers.plugins.numpyhelper import Helper

CHUNK_SIZE = 1 * 1024 * 1024  # 8 KB chunk size for reading/writing files
SPOOLED_MAX_SIZE = 10 * 1024 * 1024  # 10 MB max size for spooled temporary files


class ScaleoutModel:
    """The ScaleoutModel class is the primary model representation in the Scaleout framework.
    A ScaleoutModel object contains a data object (tempfile.SpooledTemporaryFile) that holds the model parameters.
    The model parameters dict can be extracted from the data object or be used to create a model object.
    Unpacking of the model parameters is done by the helper which needs to be provided either to the the class or
    to the method
    """

    def __init__(self):
        """Initializes a ScaleoutModel object."""
        # Using SpooledTemporaryFile to handle large model data efficiently
        # It will automatically store on disk if the data exceeds the specified size
        self._data = tempfile.SpooledTemporaryFile(SPOOLED_MAX_SIZE)
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
            new_stream = tempfile.SpooledTemporaryFile(SPOOLED_MAX_SIZE)
            while chunk := self._data.read(CHUNK_SIZE):
                new_stream.write(chunk)
            new_stream.seek(0)
            self._data.seek(0)
        return new_stream

    def get_stream_unsafe(self):
        """Returns the internal stream of the model data.

        This method is not thread-safe and should be used with caution.
        """
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

    def get_filechunk_stream(self, chunk_size=CHUNK_SIZE):
        """Returns a generator that yields chunks of the model data."""
        stream = self.get_stream()
        while chunk := stream.read(chunk_size):
            yield scaleout_msg.FileChunk(data=chunk)

    @staticmethod
    def from_model_params(model_params: dict, helper=None) -> "ScaleoutModel":
        """Creates a ScaleoutModel from model parameters."""
        model_reference = ScaleoutModel()
        model_reference.helper = helper
        if helper is None:
            # No helper provided, using numpy helper as default
            helper = Helper()
        helper.save(model_params, model_reference._data)
        model_reference._data.seek(0)
        return model_reference

    @staticmethod
    def from_file(file_path: str) -> "ScaleoutModel":
        """Creates a ScaleoutModel from a file."""
        with open(file_path, "rb") as file:
            return ScaleoutModel.from_stream(file)

    @staticmethod
    def from_stream(stream: BinaryIO) -> "ScaleoutModel":
        """Creates a ScaleoutModel from a stream."""
        model_reference = ScaleoutModel()
        while chunk := stream.read(CHUNK_SIZE):
            model_reference._data.write(chunk)
        model_reference._data.seek(0)
        return model_reference

    @staticmethod
    def from_filechunk_stream(filechunk_stream: Iterable[scaleout_msg.FileChunk]) -> "ScaleoutModel":
        """Creates a ScaleoutModel from a filechunk stream."""
        model_reference = ScaleoutModel()
        for chunk in filechunk_stream:
            if chunk.data:
                model_reference._data.write(chunk.data)
        model_reference._data.seek(0)
        return model_reference
