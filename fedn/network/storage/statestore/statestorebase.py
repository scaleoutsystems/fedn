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
    def transition(self, state):
        """Transition the statestore to a new state.

        :param state: The new state.
        :type state: str
        """
        pass

    @abstractmethod
    def set_latest_model(self, model_id):
        """Set the latest model id in the statestore.

        :param model_id: The model id.
        :type model_id: str
        """
        pass

    @abstractmethod
    def get_latest_model(self):
        """Get the latest model id from the statestore.

        :return: The model object.
        :rtype: ObjectId
        """
        pass

    @abstractmethod
    def is_inited(self):
        """Check if the statestore is initialized.

        :return: True if initialized, else False.
        :rtype: bool
        """
        pass
