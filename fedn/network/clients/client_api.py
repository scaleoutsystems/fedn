import enum
from typing import Tuple

import requests

from fedn.common.config import FEDN_AUTH_SCHEME, FEDN_CUSTOM_URL_PREFIX


# Enum for respresenting the result of connecting to the FEDn API
class ConnectToApiResult(enum.Enum):
    Assigned = 0
    UnAuthorized = 1
    UnMatchedConfig = 2
    IncorrectUrl = 3
    UnknownError = 4


class ClientAPI:
    def __init__(self):
        self._subscribers = {"train": [], "validation": []}

    def subscribe(self, event_type: str, callback: callable):
        """Subscribe to a specific event."""
        if event_type in self._subscribers:
            self._subscribers[event_type].append(callback)
        else:
            raise ValueError(f"Unsupported event type: {event_type}")

    def notify_subscribers(self, event_type: str, *args, **kwargs):
        """Notify all subscribers about a specific event."""
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                callback(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported event type: {event_type}")

    def train(self, *args, **kwargs):
        """Function to be triggered from the server via gRPC."""
        # Perform training logic here
        print("Training started with args:", args, "and kwargs:", kwargs)

        # Notify all subscribers about the train event
        self.notify_subscribers("train", *args, **kwargs)

    def validate(self, *args, **kwargs):
        """Function to be triggered from the server via gRPC."""
        # Perform validation logic here
        print("Validation started with args:", args, "and kwargs:", kwargs)

        # Notify all subscribers about the validation event
        self.notify_subscribers("validation", *args, **kwargs)

    def connect_to_api(self, url: str, token: str, json: dict) -> Tuple[ConnectToApiResult, str]:
        url_endpoint = f"{url}{FEDN_CUSTOM_URL_PREFIX}/add_client"

        response = requests.post(
            url=url_endpoint,
            json=json,
            allow_redirects=True,
            headers={"Authorization": f"{FEDN_AUTH_SCHEME} {token}"},
        )

        if response.status_code == 200:
            return ConnectToApiResult.Assigned, "Client assigned successfully"
        elif response.status_code == 401:
            return ConnectToApiResult.UnAuthorized, "Unauthorized"
        elif response.status_code == 400:
            json_response = response.json()
            return ConnectToApiResult.UnMatchedConfig, json_response["message"]
        elif response.status_code == 404:
            return ConnectToApiResult.IncorrectUrl, "Incorrect URL"

        return ConnectToApiResult.UnknownError, "Unknown error occurred"


# # Example usage
# def on_train(*args, **kwargs):
#     print("Training event received with args:", args, "and kwargs:", kwargs)


# def on_validation(*args, **kwargs):
#     print("Validation event received with args:", args, "and kwargs:", kwargs)


# client_api = ClientAPI()
# client_api.subscribe("train", on_train)
# client_api.subscribe("validation", on_validation)

# # Simulate a train event triggered from the server
# client_api.train(epochs=10, batch_size=32)

# # Simulate a validation event triggered from the server
# client_api.validate(validation_split=0.2)
