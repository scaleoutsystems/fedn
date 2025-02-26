"""Base class for artifacts repository implementations."""

import abc
from typing import IO


class RepositoryBase(abc.ABC):
    """Base class for artifacts repository implementations."""

    @abc.abstractmethod
    def set_artifact(self, instance_name: str, instance: IO, bucket: str) -> None:
        """Set object with name instance_name.

        :param instance_name: The name of the object
        :type instance_name: str
        :param instance: The object
        :type instance: Any
        :param bucket: The bucket name
        :type bucket: str
        """
        raise NotImplementedError("Must be implemented by subclass")

    @abc.abstractmethod
    def get_artifact(self, instance_name: str, bucket: str) -> IO:
        """Retrieve object with name instance_name.

        :param instance_name: The name of the object to retrieve
        :type instance_name: str
        :param bucket: The bucket name
        :type bucket: str
        :return: The retrieved object
        :rtype: Any
        """
        raise NotImplementedError("Must be implemented by subclass")

    @abc.abstractmethod
    def get_artifact_stream(self, instance_name: str, bucket: str) -> IO:
        """Return a stream handler for object with name instance_name.

        :param instance_name: The name of the object
        :type instance_name: str
        :param bucket: The bucket name
        :type bucket: str
        :return: Stream handler for object instance_name
        :rtype: IO
        """
        raise NotImplementedError("Must be implemented by subclass")
