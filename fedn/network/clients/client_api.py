import enum
import os
import time
from typing import Any, Tuple

import requests

from fedn.common.config import FEDN_AUTH_SCHEME, FEDN_CUSTOM_URL_PREFIX, FEDN_PACKAGE_EXTRACT_DIR
from fedn.common.log_config import logger
from fedn.network.clients.package_runtime import PackageRuntime


# Enum for respresenting the result of connecting to the FEDn API
class ConnectToApiResult(enum.Enum):
    Assigned = 0
    UnAuthorized = 1
    UnMatchedConfig = 2
    IncorrectUrl = 3
    UnknownError = 4


def get_compute_package_dir_path():
    result = None

    if FEDN_PACKAGE_EXTRACT_DIR:
        result = os.path.join(os.getcwd(), FEDN_PACKAGE_EXTRACT_DIR)
    else:
        dirname = +"compute-package-" + time.strftime("%Y%m%d-%H%M%S")
        result = os.path.join(os.getcwd(), dirname)

    if not os.path.exists(result):
        os.mkdir(result)

    return result


class ClientAPI:
    def __init__(self):
        self._subscribers = {"train": [], "validation": []}
        path = get_compute_package_dir_path()
        self._package_runtime = PackageRuntime(path)

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
        logger.info("Training started with args:", args, "and kwargs:", kwargs)

        # Notify all subscribers about the train event
        self.notify_subscribers("train", *args, **kwargs)

    def validate(self, *args, **kwargs):
        """Function to be triggered from the server via gRPC."""
        # Perform validation logic here
        logger.info("Validation started with args:", args, "and kwargs:", kwargs)

        # Notify all subscribers about the validation event
        self.notify_subscribers("validation", *args, **kwargs)

    def connect_to_api(self, url: str, token: str, json: dict) -> Tuple[ConnectToApiResult, Any]:
        # TODO: Use new API endpoint (v1)
        url_endpoint = f"{url}/add_client"

        try:
            response = requests.post(
                url=url_endpoint,
                json=json,
                allow_redirects=True,
                headers={"Authorization": f"{FEDN_AUTH_SCHEME} {token}"},
            )

            if response.status_code == 200:
                json_response = response.json()
                return ConnectToApiResult.Assigned, json_response
            elif response.status_code == 401:
                return ConnectToApiResult.UnAuthorized, "Unauthorized"
            elif response.status_code == 400:
                json_response = response.json()
                return ConnectToApiResult.UnMatchedConfig, json_response["message"]
            elif response.status_code == 404:
                return ConnectToApiResult.IncorrectUrl, "Incorrect URL"
        except Exception:
            pass

        return ConnectToApiResult.UnknownError, "Unknown error occurred"

    def download_compute_package(self, url: str, token: str, name: str = None) -> bool:
        """Download compute package from controller
        :param host: host of controller
        :param port: port of controller
        :param token: token for authentication
        :param name: name of package
        :return: True if download was successful, None otherwise
        :rtype: bool
        """
        return self._package_runtime.download_compute_package(url, token, name)

    def set_compute_package_checksum(self, url: str, token: str, name: str = None) -> bool:
        """Get checksum of compute package from controller
        :param host: host of controller
        :param port: port of controller
        :param token: token for authentication
        :param name: name of package
        :return: checksum of the compute package
        :rtype: str
        """
        return self._package_runtime.set_checksum(url, token, name)

    def unpack_compute_package(self):
        result, path = self._package_runtime.unpack_compute_package()
        if result:
            logger.info(f"Compute package unpacked to: {path}")
        else:
            logger.info("Error: Could not unpack compute package")


# # Example usage
# def on_train(*args, **kwargs):
#     logger.info("Training event received with args:", args, "and kwargs:", kwargs)


# def on_validation(*args, **kwargs):
#     logger.info("Validation event received with args:", args, "and kwargs:", kwargs)


# client_api = ClientAPI()
# client_api.subscribe("train", on_train)
# client_api.subscribe("validation", on_validation)

# # Simulate a train event triggered from the server
# client_api.train(epochs=10, batch_size=32)

# # Simulate a validation event triggered from the server
# client_api.validate(validation_split=0.2)
