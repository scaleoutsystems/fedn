import collections
from abc import ABC,abstractmethod

class HelperBase(ABC):
    """ Abstract class defining helpers. """
    def __init__(self):
        """ fdfhjskhf """
    @abstractmethod
    def increment_average(self,model,model_next,n):
        """ fdsfsdf """
        pass

    @abstractmethod
    def save_model(self,model,path):
        """ """
        pass

    @abstractmethod
    def load_model(self, model):
        """ """
        pass

