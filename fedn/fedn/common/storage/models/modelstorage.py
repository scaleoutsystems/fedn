

from abc import ABC, abstractmethod


class ModelStorage(ABC):


    def get(self, model_id):
        pass

    def set(self, model_id, model_metadata, model=None):
        pass

