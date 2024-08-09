import abc


class RepositoryBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def set_artifact(self, instance_name, instance, bucket):
        """Set object with name object_name

        :param instance_name: The name of the object
        :tyep insance_name: str
        :param instance: the object
        :param bucket: The bucket name
        :type bucket: str
        """
        raise NotImplementedError("Must be implemented by subclass")

    @abc.abstractmethod
    def get_artifact(self, instance_name, bucket):
        """Retrive object with name instance_name.

        :param instance_name: The name of the object to retrieve
        :type instance_name: str
        :param bucket: The bucket name
        :type bucket: str
        """
        raise NotImplementedError("Must be implemented by subclass")

    @abc.abstractmethod
    def get_artifact_stream(self, instance_name, bucket):
        """Return a stream handler for object with name instance_name.

        :param instance_name: The name if the object
        :type instance_name: str
        :param bucket: The bucket name
        :type bucket: str
        :return: stream handler for object instance name
        """
        raise NotImplementedError("Must be implemented by subclass")
