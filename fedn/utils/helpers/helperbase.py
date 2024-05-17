from abc import ABC, abstractmethod


class HelperBase(ABC):
    """Abstract class defining helpers."""

    def __init__(self):
        """Initialize helper."""
        self.name = self.__class__.__name__

    @abstractmethod
    def increment_average(self, m1, m2, a, W):
        """Compute one increment of incremental weighted averaging.

        :param m1: Current model weights in array-like format.
        :param m2: New model weights in array-like format.
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
        """Load weights from file or filelike.

        :param fh: file path, filehandle, filelike.
        :return: Weights in array-like format.
        """
        pass
