

from abc import ABC, abstractmethod


class ModelStorage(ABC):

    @abstractmethod
    def exist(self, model_id):
        pass

    @abstractmethod
    def get(self, model_id):
        pass

#    @abstractmethod
#    def set(self, model_id, model):
#        pass

    @abstractmethod
    def get_meta(self, model_id):
        pass

    @abstractmethod
    def set_meta(self, model_id, model_metadata):
        pass