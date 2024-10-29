import enum
import os
import time
from io import BytesIO
from typing import Any, Tuple

import requests

import fedn.network.grpc.fedn_pb2 as fedn
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
        logger.info("Training started")

        # Notify all subscribers about the train event
        self.notify_subscribers("train", *args, **kwargs)

    def validate(self, *args, **kwargs):
        """Function to be triggered from the server via gRPC."""
        # Perform validation logic here
        logger.info("Validation started")

        # Notify all subscribers about the validation event
        self.notify_subscribers("validation", *args, **kwargs)

    def connect_to_api(self, url: str, token: str, json: dict) -> Tuple[ConnectToApiResult, Any]:
        # TODO: Use new API endpoint (v1)
        url_endpoint = f"{url}add_client"
        logger.info(f"Connecting to API endpoint: {url_endpoint}")

        try:
            response = requests.post(
                url=url_endpoint,
                json=json,
                allow_redirects=True,
                headers={"Authorization": f"{FEDN_AUTH_SCHEME} {token}"},
            )

            if response.status_code == 200:
                logger.info("Connect to FEDn Api - Client assinged to controller")
                json_response = response.json()
                return ConnectToApiResult.Assigned, json_response
            elif response.status_code == 203:
                json_response = response.json()
                logger.info("Connect to FEDn Api - Remote compute package missing.")
                return ConnectToApiResult.ComputePackgeMissing, json_response
            elif response.status_code == 401:
                logger.warning("Connect to FEDn Api - Unauthorized")
                return ConnectToApiResult.UnAuthorized, "Unauthorized"
            elif response.status_code == 400:
                json_response = response.json()
                msg = json_response["message"]
                logger.warning(f"Connect to FEDn Api - {msg}")
                return ConnectToApiResult.UnMatchedConfig, msg
            elif response.status_code == 404:
                logger.warning("Connect to FEDn Api - Incorrect URL")
                return ConnectToApiResult.IncorrectUrl, "Incorrect URL"
        except Exception:
            pass

        logger.warning("Connect to FEDn Api - Unknown error occurred")
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
            return True
        else:
            logger.error("Error: Could not set dispatcher")
            return False

    def get_or_set_environment(self) -> bool:
        try:
            logger.info("Initiating Dispatcher with entrypoint set to: startup")
            activate_cmd = self.dispatcher._get_or_create_python_env()
            self.dispatcher.run_cmd("startup")
        except KeyError:
            logger.info("No startup command found in package. Continuing.")
            return False
        except Exception as e:
            logger.error(f"Caught exception: {type(e).__name__}")
            return False

        if activate_cmd:
            logger.info("To activate the virtual environment, run: {}".format(activate_cmd))

        return True

    # GRPC functions
    def init_grpchandler(self, config: GrpcConnectionOptions, client_name: str, token: str):
        try:
            if "fqdn" in config and config["fqdn"] and len(config["fqdn"]) > 0:
                host = config["fqdn"]
                port = 443
            else:
                host = config["host"]
                port = config["port"]
            combiner_name = config["host"]

            self.grpc_handler = GrpcHandler(host=host, port=port, name=client_name, token=token, combiner_name=combiner_name)

            logger.info("Successfully initialized GRPC connection")
            return True
        except Exception:
            logger.error("Error: Could not initialize GRPC connection")
            return False


    def send_heartbeats(self, client_name: str, client_id: str, update_frequency: float = 2.0):
        self.grpc_handler.send_heartbeats(client_name=client_name, client_id=client_id, update_frequency=update_frequency)

    def listen_to_task_stream(self, client_name: str, client_id: str):
        self.grpc_handler.listen_to_task_stream(client_name=client_name, client_id=client_id, callback=self._task_stream_callback)

    def _task_stream_callback(self, request):
        if request.type == fedn.StatusType.MODEL_UPDATE:
            self.train(request)
        elif request.type == fedn.StatusType.MODEL_VALIDATION:
            self.validate(request)

    def get_model_from_combiner(self, id: str, client_name: str, timeout: int = 20) -> BytesIO:
        return self.grpc_handler.get_model_from_combiner(id=id, client_name=client_name, timeout=timeout)

    def send_model_to_combiner(self, model: BytesIO, id: str):
        return self.grpc_handler.send_model_to_combiner(model, id)

    def send_status(self, msg: str, log_level=fedn.Status.INFO, type=None, request=None, sesssion_id: str = None, sender_name: str = None):
        return self.grpc_handler.send_status(msg, log_level, type, request, sesssion_id, sender_name)

    def send_model_update(self,
        sender_name: str,
        sender_role: fedn.Role,
        client_id: str,
        model_id: str,
        model_update_id: str,
        receiver_name: str,
        receiver_role: fedn.Role,
        meta: dict
    ) -> bool:
        return self.grpc_handler.send_model_update(
            sender_name=sender_name,
            sender_role=sender_role,
            client_id=client_id,
            model_id=model_id,
            model_update_id=model_update_id,
            receiver_name=receiver_name,
            receiver_role=receiver_role,
            meta=meta
        )

    def send_model_validation(self,
        sender_name: str,
        receiver_name: str,
        receiver_role: fedn.Role,
        model_id: str,
        metrics: dict,
        correlation_id: str,
        session_id: str
    ) -> bool:
        return self.grpc_handler.send_model_validation(sender_name, receiver_name, receiver_role, model_id, metrics, correlation_id, session_id)

    # Init functions
    def init_remote_compute_package(self, url: str, token: str, package_checksum: str = None) -> bool:
        result: bool = self.download_compute_package(url, token)
        if not result:
            logger.error("Could not download compute package")
            return False
        result: bool = self.set_compute_package_checksum(url, token)
        if not result:
            logger.error("Could not set checksum")
            return False

        if package_checksum:
            result: bool = self.validate_compute_package(package_checksum)
            if not result:
                logger.error("Could not validate compute package")
                return False

        result, path = self.unpack_compute_package()

        if not result:
            logger.error("Could not unpack compute package")
            return False

        logger.info(f"Compute package unpacked to: {path}")

        result = self.set_dispatcher(path)

        if not result:
            logger.error("Could not set dispatcher")
            return False

        logger.info("Dispatcher set")

        result = self.get_or_set_environment()

        if not result:
            logger.error("Could not set environment")
            return False

        return True

    def init_local_compute_package(self):
        path = os.path.join(os.getcwd(), "client")
        result = self.set_dispatcher(path)

        if not result:
            logger.error("Could not set dispatcher")
            return False

        result = self.get_or_set_environment()

        if not result:
            logger.error("Could not set environment")
            return False

        logger.info("Dispatcher set")

        return True
