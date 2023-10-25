from abc import ABC, abstractmethod


class StateStoreBase(ABC):
    """

    """

    def __init__(self):
        pass

    @abstractmethod
    def state(self):
        """

        """
        pass

    @abstractmethod
    def transition(self, state):
        """

        :param state:
        """
        pass

    @abstractmethod
    def set_latest(self, model_id):
        """

        :param model_id:
        """
        pass

    @abstractmethod
    def get_latest(self):
        """

        """
        pass

    @abstractmethod
    def is_inited(self):
        """

        """
        pass
