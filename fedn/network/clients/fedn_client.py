"""FednClient class for interacting with the FEDn network."""

import enum
import json
import os
import threading
import time
import uuid
from io import BytesIO
from typing import Any, Optional, Tuple

import requests

import fedn.network.grpc.fedn_pb2 as fedn
from fedn.common.config import FEDN_AUTH_SCHEME, FEDN_PACKAGE_EXTRACT_DIR
from fedn.common.log_config import logger
from fedn.network.clients.grpc_handler import GrpcHandler
from fedn.network.clients.package_runtime import PackageRuntime
from fedn.utils.dispatcher import Dispatcher

# Constants for HTTP status codes
HTTP_STATUS_OK = 200
HTTP_STATUS_NO_CONTENT = 204
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_NOT_FOUND = 404
HTTP_STATUS_PACKAGE_MISSING = 203

# Default timeout for requests
REQUEST_TIMEOUT = 10  # seconds


class GrpcConnectionOptions:
    """Options for configuring the GRPC connection."""

    def __init__(self, status: str, host: str, fqdn: str, package: str, ip: str, port: int, helper_type: str) -> None:
        """Initialize GrpcConnectionOptions."""
        self.status = status
        self.host = host
        self.fqdn = fqdn
        self.package = package
        self.ip = ip
        self.port = port
        self.helper_type = helper_type

    @classmethod
    def from_dict(cls, config: dict) -> "GrpcConnectionOptions":
        """Create a GrpcConnectionOptions instance from a JSON string."""
        return cls(
            status=config.get("status", ""),
            host=config.get("host", ""),
            fqdn=config.get("fqdn", ""),
            package=config.get("package", ""),
            ip=config.get("ip", ""),
            port=config.get("port", 0),
            helper_type=config.get("helper_type", ""),
        )


class ConnectToApiResult(enum.Enum):
    """Enum for representing the result of connecting to the FEDn API."""

    Assigned = 0
    ComputePackageMissing = 1
    UnAuthorized = 2
    UnMatchedConfig = 3
    IncorrectUrl = 4
    UnknownError = 5


def get_compute_package_dir_path() -> str:
    """Get the directory path for the compute package."""
    if FEDN_PACKAGE_EXTRACT_DIR:
        result = os.path.join(os.getcwd(), FEDN_PACKAGE_EXTRACT_DIR)
    else:
        dirname = "compute-package-" + time.strftime("%Y%m%d-%H%M%S")
        result = os.path.join(os.getcwd(), dirname)

    if not os.path.exists(result):
        os.mkdir(result)

    return result


class FednClient:
    """Client for interacting with the FEDn network."""

    def __init__(
        self, train_callback: Optional[callable] = None, validate_callback: Optional[callable] = None, predict_callback: Optional[callable] = None
    ) -> None:
        """Initialize the FednClient."""
        self.train_callback = train_callback
        self.validate_callback = validate_callback
        self.predict_callback = predict_callback

        path = get_compute_package_dir_path()
        self._package_runtime = PackageRuntime(path)

        self.dispatcher: Optional[Dispatcher] = None
        self.grpc_handler: Optional[GrpcHandler] = None

    def set_train_callback(self, callback: callable) -> None:
        """Set the train callback."""
        self.train_callback = callback

    def set_validate_callback(self, callback: callable) -> None:
        """Set the validate callback."""
        self.validate_callback = callback

    def set_predict_callback(self, callback: callable) -> None:
        """Set the predict callback."""
        self.predict_callback = callback

    def connect_to_api(self, url: str, token: str, json: dict) -> Tuple[ConnectToApiResult, Any]:
        """Connect to the FEDn API."""
        url_endpoint = f"{url}api/v1/clients/add"
        logger.info(f"Connecting to API endpoint: {url_endpoint}")

        try:
            response = requests.post(
                url=url_endpoint,
                json=json,
                allow_redirects=True,
                headers={"Authorization": f"{FEDN_AUTH_SCHEME} {token}"},
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == HTTP_STATUS_OK:
                logger.info("Connect to FEDn Api - Client assigned to controller")
                json_response = response.json()
                combiner_config = GrpcConnectionOptions.from_dict(json_response)
                return ConnectToApiResult.Assigned, combiner_config

            if response.status_code == HTTP_STATUS_PACKAGE_MISSING:
                json_response = response.json()
                logger.info("Connect to FEDn Api - Remote compute package missing.")
                return ConnectToApiResult.ComputePackageMissing, json_response

            if response.status_code == HTTP_STATUS_UNAUTHORIZED:
                logger.warning("Connect to FEDn Api - Unauthorized")
                return ConnectToApiResult.UnAuthorized, "Unauthorized"

            if response.status_code == HTTP_STATUS_BAD_REQUEST:
                json_response = response.json()
                msg = json_response["message"]
                logger.warning(f"Connect to FEDn Api - {msg}")
                return ConnectToApiResult.UnMatchedConfig, msg

            if response.status_code == HTTP_STATUS_NOT_FOUND:
                logger.warning("Connect to FEDn Api - Incorrect URL")
                return ConnectToApiResult.IncorrectUrl, "Incorrect URL"

        except Exception as e:
            logger.warning(f"Connect to FEDn Api - Error occurred: {str(e)}")
            return ConnectToApiResult.UnknownError, str(e)

    def download_compute_package(self, url: str, token: str, name: Optional[str] = None) -> bool:
        """Download compute package from controller."""
        return self._package_runtime.download_compute_package(url, token, name)

    def set_compute_package_checksum(self, url: str, token: str, name: Optional[str] = None) -> bool:
        """Get checksum of compute package from controller."""
        return self._package_runtime.set_checksum(url, token, name)

    def unpack_compute_package(self) -> Tuple[bool, str]:
        """Unpack the compute package."""
        result, path = self._package_runtime.unpack_compute_package()
        if result:
            logger.info(f"Compute package unpacked to: {path}")
        else:
            logger.error("Error: Could not unpack compute package")

        return result, path

    def validate_compute_package(self, checksum: str) -> bool:
        """Validate the compute package."""
        return self._package_runtime.validate(checksum)

    def set_dispatcher(self, path: str) -> bool:
        """Set the dispatcher."""
        result = self._package_runtime.get_dispatcher(path)
        if result:
            self.dispatcher = result
            return True

        logger.error("Error: Could not set dispatcher")
        return False

    def get_or_set_environment(self) -> bool:
        """Get or set the environment."""
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
            logger.info(f"To activate the virtual environment, run: {activate_cmd}")

        return True

    def init_grpchandler(self, config: GrpcConnectionOptions, client_name: str, token: str) -> bool:
        """Initialize the GRPC handler."""
        try:
            if config.fqdn and len(config.fqdn) > 0:
                host = config.fqdn
                port = 443
            else:
                host = config.host
                port = config.port
            combiner_name = config.host

            self.grpc_handler = GrpcHandler(host=host, port=port, name=client_name, token=token, combiner_name=combiner_name)

            logger.info("Successfully initialized GRPC connection")
            return True
        except Exception as e:
            logger.error(f"Could not initialize GRPC connection: {e}")
            return False

    def send_heartbeats(self, client_name: str, client_id: str, update_frequency: float = 2.0) -> None:
        """Send heartbeats to the server."""
        self.grpc_handler.send_heartbeats(client_name=client_name, client_id=client_id, update_frequency=update_frequency)

    def listen_to_task_stream(self, client_name: str, client_id: str) -> None:
        """Listen to the task stream."""
        self.grpc_handler.listen_to_task_stream(client_name=client_name, client_id=client_id, callback=self._task_stream_callback)

    def _task_stream_callback(self, request: fedn.TaskRequest) -> None:
        """Handle task stream callbacks."""
        if request.type == fedn.StatusType.MODEL_UPDATE:
            self.update_local_model(request)
        elif request.type == fedn.StatusType.MODEL_VALIDATION:
            self.validate_global_model(request)
        elif request.type == fedn.StatusType.MODEL_PREDICTION:
            self.predict_global_model(request)

    def update_local_model(self, request: fedn.TaskRequest) -> None:
        """Update the local model."""
        model_id = request.model_id
        model_update_id = str(uuid.uuid4())

        tic = time.time()
        in_model = self.get_model_from_combiner(id=model_id, client_id=self.client_id)

        if in_model is None:
            logger.error("Could not retrieve model from combiner. Aborting training request.")
            return

        fetch_model_time = time.time() - tic

        if not self.train_callback:
            logger.error("No train callback set")
            return

        self.send_status(
            f"\t Starting processing of training request for model_id {model_id}",
            sesssion_id=request.session_id,
            sender_name=self.name,
            log_level=fedn.LogLevel.INFO,
            type=fedn.StatusType.MODEL_UPDATE,
        )

        logger.info(f"Running train callback with model ID: {model_id}")
        client_settings = json.loads(request.data).get("client_settings", {})
        tic = time.time()
        out_model, meta = self.train_callback(in_model, client_settings)
        meta["processing_time"] = time.time() - tic

        tic = time.time()
        self.send_model_to_combiner(model=out_model, id=model_update_id)
        meta["upload_model"] = time.time() - tic

        meta["fetch_model"] = fetch_model_time
        meta["config"] = request.data

        update = self.create_update_message(model_id=model_id, model_update_id=model_update_id, meta=meta, request=request)

        self.send_model_update(update)

        self.send_status(
            "Model update completed.",
            log_level=fedn.LogLevel.AUDIT,
            type=fedn.StatusType.MODEL_UPDATE,
            request=update,
            sesssion_id=request.session_id,
            sender_name=self.name,
        )

    def validate_global_model(self, request: fedn.TaskRequest) -> None:
        """Validate the global model."""
        model_id = request.model_id

        self.send_status(
            f"Processing validate request for model_id {model_id}",
            sesssion_id=request.session_id,
            sender_name=self.name,
            log_level=fedn.LogLevel.INFO,
            type=fedn.StatusType.MODEL_VALIDATION,
        )

        in_model = self.get_model_from_combiner(id=model_id, client_id=self.client_id)

        if in_model is None:
            logger.error("Could not retrieve model from combiner. Aborting validation request.")
            return

        if not self.validate_callback:
            logger.error("No validate callback set")
            return

        logger.info(f"Running validate callback with model ID: {model_id}")
        metrics = self.validate_callback(in_model)

        if metrics is not None:
            # Send validation
            validation = self.create_validation_message(metrics=metrics, request=request)

            result: bool = self.send_model_validation(validation)

            if result:
                self.send_status(
                    "Model validation completed.",
                    log_level=fedn.LogLevel.AUDIT,
                    type=fedn.StatusType.MODEL_VALIDATION,
                    request=validation,
                    sesssion_id=request.session_id,
                    sender_name=self.name,
                )
            else:
                self.send_status(
                    f"Client {self.name} failed to complete model validation.",
                    log_level=fedn.LogLevel.WARNING,
                    request=request,
                    sesssion_id=request.session_id,
                    sender_name=self.name,
                )

    def predict_global_model(self, request: fedn.TaskRequest) -> None:
        """Predict using the global model."""
        model_id = request.model_id
        model = self.get_model_from_combiner(id=model_id, client_id=self.client_id)

        if model is None:
            logger.error("Could not retrieve model from combiner. Aborting prediction request.")
            return

        if not self.predict_callback:
            logger.error("No predict callback set")
            return

        logger.info(f"Running predict callback with model ID: {model_id}")
        prediction = self.predict_callback(model)

        prediction_message = self.create_prediction_message(prediction=prediction, request=request)

        self.send_model_prediction(prediction_message)

    def create_update_message(self, model_id: str, model_update_id: str, meta: dict, request: fedn.TaskRequest) -> fedn.ModelUpdate:
        """Create an update message."""
        return self.grpc_handler.create_update_message(
            sender_name=self.name,
            model_id=model_id,
            model_update_id=model_update_id,
            receiver_name=request.sender.name,
            receiver_role=request.sender.role,
            meta=meta,
        )

    def create_validation_message(self, metrics: dict, request: fedn.TaskRequest) -> fedn.ModelValidation:
        """Create a validation message."""
        return self.grpc_handler.create_validation_message(
            sender_name=self.name,
            receiver_name=request.sender.name,
            receiver_role=request.sender.role,
            model_id=request.model_id,
            metrics=json.dumps(metrics),
            correlation_id=request.correlation_id,
            session_id=request.session_id,
        )

    def create_prediction_message(self, prediction: dict, request: fedn.TaskRequest) -> fedn.ModelPrediction:
        """Create a prediction message."""
        return self.grpc_handler.create_prediction_message(
            sender_name=self.name,
            receiver_name=request.sender.name,
            receiver_role=request.sender.role,
            model_id=request.model_id,
            prediction_output=json.dumps(prediction),
            correlation_id=request.correlation_id,
            session_id=request.session_id,
        )

    def set_name(self, name: str) -> None:
        """Set the client name."""
        logger.info(f"Setting client name to: {name}")
        self.name = name

    def set_client_id(self, client_id: str) -> None:
        """Set the client ID."""
        logger.info(f"Setting client ID to: {client_id}")
        self.client_id = client_id

    def run(self) -> None:
        """Run the client."""
        threading.Thread(target=self.send_heartbeats, kwargs={"client_name": self.name, "client_id": self.client_id}, daemon=True).start()
        try:
            self.listen_to_task_stream(client_name=self.name, client_id=self.client_id)
        except KeyboardInterrupt:
            logger.info("Client stopped by user.")

    def get_model_from_combiner(self, id: str, client_id: str, timeout: int = 20) -> BytesIO:
        """Get the model from the combiner."""
        return self.grpc_handler.get_model_from_combiner(id=id, client_id=client_id, timeout=timeout)

    def send_model_to_combiner(self, model: BytesIO, id: str) -> None:
        """Send the model to the combiner."""
        self.grpc_handler.send_model_to_combiner(model, id)

    def send_status(
        self,
        msg: str,
        log_level: fedn.LogLevel = fedn.LogLevel.INFO,
        type: Optional[fedn.StatusType] = None,
        request: Optional[fedn.TaskRequest] = None,
        sesssion_id: Optional[str] = None,
        sender_name: Optional[str] = None,
    ) -> None:
        """Send the status."""
        self.grpc_handler.send_status(msg, log_level, type, request, sesssion_id, sender_name)

    def send_model_update(self, update: fedn.ModelUpdate) -> bool:
        """Send the model update."""
        return self.grpc_handler.send_model_update(update)

    def send_model_validation(self, validation: fedn.ModelValidation) -> bool:
        """Send the model validation."""
        return self.grpc_handler.send_model_validation(validation)

    def send_model_prediction(self, prediction: fedn.ModelPrediction) -> bool:
        """Send the model prediction."""
        return self.grpc_handler.send_model_prediction(prediction)

    def init_remote_compute_package(self, url: str, token: str, package_checksum: Optional[str] = None) -> bool:
        """Initialize the remote compute package."""
        result = self.download_compute_package(url, token)
        if not result:
            logger.error("Could not download compute package")
            return False
        result = self.set_compute_package_checksum(url, token)
        if not result:
            logger.error("Could not set checksum")
            return False

        if package_checksum:
            result = self.validate_compute_package(package_checksum)
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

        return True

    def init_local_compute_package(self) -> bool:
        """Initialize the local compute package."""
        path = os.path.join(os.getcwd(), "client")
        result = self.set_dispatcher(path)

        if not result:
            logger.error("Could not set dispatcher")
            return False

        result = self.get_or_set_environment()

        logger.info("Dispatcher set")

        return True
