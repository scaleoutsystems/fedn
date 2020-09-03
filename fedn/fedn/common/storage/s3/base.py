import abc


class Repository(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def set_artifact(self, instance_name, instance):
        raise NotImplementedError("Must be implemented by subclass")

    @abc.abstractmethod
    def get_artifact(self, instance_name):
        raise NotImplementedError("Must be implemented by subclass")
