import enum
import os
import threading
import time
from typing import Any, Tuple

import requests

from fedn.common.config import FEDN_AUTH_SCHEME, FEDN_PACKAGE_EXTRACT_DIR
from fedn.common.log_config import logger
from fedn.network.clients.grpc_handler import GrpcHandler
from fedn.network.clients.package_runtime import PackageRuntime
from fedn.utils.dispatcher import Dispatcher


class GrpcConnectionOptions:
    def __init__(self, status: str, host: str, fqdn: str, package: str, ip: str, port: int, helper_type: str):
        self.status = status
        self.host = host
        self.fqdn = fqdn
        self.package = package
        self.ip = ip
        self.port = port
        self.helper_type = helper_type


# Enum for respresenting the result of connecting to the FEDn API
class ConnectToApiResult(enum.Enum):
    Assigned = 0
    ComputePackgeMissing = 1
    UnAuthorized = 2
    UnMatchedConfig = 3
    IncorrectUrl = 4
    UnknownError = 5


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

        self.dispatcher: Dispatcher = None
        self.grpc_handler: GrpcHandler = None

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

            # TODO: If 203 it should try again...
            if response.status_code in [200]:
                json_response = response.json()
                return ConnectToApiResult.Assigned, json_response
            elif response.status_code == 203:
                json_response = response.json()
                return ConnectToApiResult.ComputePackgeMissing, json_response
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

    def unpack_compute_package(self) -> Tuple[bool, str]:
        result, path = self._package_runtime.unpack_compute_package()
        if result:
            logger.info(f"Compute package unpacked to: {path}")
        else:
            logger.error("Error: Could not unpack compute package")

        return result, path

    def validate_compute_package(self, checksum: str) -> bool:
        return self._package_runtime.validate(checksum)

    def set_dispatcher(self, path) -> bool:
        result = self._package_runtime.get_dispatcher(path)
        if result:
            self.dispatcher = result
            logger.info("Dispatcher set")
            return True
        else:
            logger.error("Error: Could not set dispatcher")
            return False

    def get_or_set_environment(self, path: str) -> str:
        activate_cmd = self.dispatcher._get_or_create_python_env()
        if activate_cmd:
            logger.info("To activate the virtual environment, run: {}".format(activate_cmd))

    # def _subscribe_to_combiner(self, config):
    #     """Listen to combiner message stream and start all processing threads.

    #     :param config: A configuration dictionary containing connection information for
    #     | the discovery service (controller) and settings governing e.g.
    #     | client-combiner assignment behavior.
    #     """
    #     # Start sending heartbeats to the combiner.
    #     threading.Thread(target=self._send_heartbeat, kwargs={"update_frequency": config["heartbeat_interval"]}, daemon=True).start()

    #     # Start listening for combiner training and validation messages
    #     threading.Thread(target=self._listen_to_task_stream, daemon=True).start()
    #     self._connected = True

    #     # Start processing the client message inbox
    #     threading.Thread(target=self.process_request, daemon=True).start()

    def init_grpchandler(self, config: GrpcConnectionOptions, client_name: str, token: str):
        try:
            if config["fqdn"] and len(config["fqdn"]) > 0:
                host = config["fqdn"]
                port = 443
            else:
                host = config["host"]
                port = config["port"]
            combiner_name = config["host"]

            self.grpc_handler = GrpcHandler(host=host, port=port, name=client_name, token=token, combiner_name=combiner_name)

            return True
        except Exception:
            logger.error("Error: Could not initialize GRPC handler")
            return False


    def send_heartbeats(self, client_name: str, client_id: str, update_frequency: float = 2.0):
        self.grpc_handler.send_heartbeats(client_name=client_name, client_id=client_id, update_frequency=update_frequency)



