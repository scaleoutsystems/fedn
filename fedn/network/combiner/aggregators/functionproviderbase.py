from abc import ABC, ABCMeta, abstractmethod

import numpy as np


class NamingEnforcementMeta(type):
    """Metaclass that enforces the name 'FunctionProvider' for any class
    inheriting from FunctionProviderBase.
    """

    def __init__(cls, name, bases, attrs):
        if name != "FunctionProvider" and name != "FunctionProviderBase":  # Allow the base class name
            raise TypeError(f"Class {name} must be named 'FunctionProvider'")
        super().__init__(name, bases, attrs)


class CombinedMeta(NamingEnforcementMeta, ABCMeta):
    """A metaclass that combines ABCMeta and NamingEnforcementMeta to allow
    the use of both abstract base classes and custom naming enforcement.
    """

    pass


class FunctionProviderBase(ABC, metaclass=CombinedMeta):
    """Abstract base class that defines the structure for function providers.
    Enforces the implementation of certain methods and provides shared functionality.
    """

    def __init__(self) -> None:
        """Initialize the FunctionProviderBase class. This method can be overridden
        by subclasses if initialization logic is required.
        """
        pass

    @abstractmethod
    def aggregate(self, parameters: list[list[np.ndarray]]) -> list[np.ndarray]:
        """Aggregates a list of parameters from clients.

        Args:
        ----
            parameters (list[list[np.ndarray]]): A list where each element is a list
            of numpy arrays representing parameters from a client.

        Returns:
        -------
            list[np.ndarray]: A list of numpy arrays representing the aggregated
            parameters across all clients.

        """
        pass

    def get_model_metadata(self) -> dict:
        """Returns metadata related to the model, which gets distributed to the clients.
        The dictionary may only contain primitive types.


        Returns
        -------
            dict: A dictionary containing metadata information.

        """
        return {}
