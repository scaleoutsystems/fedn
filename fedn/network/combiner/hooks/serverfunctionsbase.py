from abc import ABC
from typing import Dict, List, Tuple

import numpy as np


class ServerFunctionsBase(ABC):
    """Base class that defines the structure for the Server Functions. Override these functions
    to add to the server workflow.
    """

    def __init__(self) -> None:
        """Initialize the ServerFunctionsBase class. This method can be overridden
        by subclasses if initialization logic is required.
        """
        pass

    def aggregate(self, previous_global: List[np.ndarray], client_updates: Dict[str, Tuple[List[np.ndarray], Dict]]) -> List[np.ndarray]:
        """Aggregates a list of parameters from clients.

        Args:
        ----
            previous_global (list[np.ndarray]): A list of parameters representing the global
            model from the previous round.

            client_updates (list[list[np.ndarray]]): A dictionary where the key is client ID,
            pointing to a tuple with the first element being client parameter and second element
            being the clients metadata.

        Returns:
        -------
            list[np.ndarray]: A list of numpy arrays representing the aggregated
            parameters across all clients.

        """
        pass

    def client_settings(self, global_model: List[np.ndarray]) -> Dict:
        """Returns metadata related to the model, which gets distributed to the clients.
        The dictionary may only contain primitive types.

        Args:
        ----
            global_model (list[np.ndarray]): A list of parameters representing the global
            model for the upcomming round.

        Returns:
        -------
            dict: A dictionary containing metadata information, supporting only primitive python types.

        """
        pass

    def client_selection(self, client_ids: List[str]) -> List:
        """Returns a list of client_id's of which clients to be used for the next training request.

        Args:
        ----
            client_ids (list[str]): A list of client_ids for all connected clients.

        Returns:
        -------
            list[str]: A list of client ids for which clients should be chosen for the next training round.

        """
        pass


# base implementation
class ServerFunctions(ServerFunctionsBase):
    pass
