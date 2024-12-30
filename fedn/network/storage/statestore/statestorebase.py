from abc import ABC, abstractmethod


class StateStoreBase(ABC):
    """ """

    def __init__(self):
        pass

    @abstractmethod
    def state(self):
        """Return the current state of the statestore."""
        pass

    @abstractmethod
    def is_inited(self):
        """Check if the statestore is initialized.

        :return: True if initialized, else False.
        :rtype: bool
        """
        pass
