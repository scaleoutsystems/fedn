"""FednClient class for interacting with the FEDn network."""

import enum
import json
import threading
import time
import uuid
from contextlib import contextmanager
from io import BytesIO
from typing import Any, Optional, Tuple, Union
from urllib.parse import urljoin

import psutil
import requests

import fedn.network.grpc.fedn_pb2 as fedn
from fedn.common.config import FEDN_AUTH_SCHEME, FEDN_CONNECT_API_SECURE
from fedn.common.log_config import logger
from fedn.network.clients.grpc_handler import GrpcConnectionOptions, GrpcHandler, RetryException
from fedn.network.clients.http_status_codes import (
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_NOT_FOUND,
    HTTP_STATUS_OK,
    HTTP_STATUS_PACKAGE_MISSING,
    HTTP_STATUS_UNAUTHORIZED,
)
from fedn.network.clients.logging_context import LoggingContext

# Default timeout for requests
REQUEST_TIMEOUT = 10  # seconds


class ConnectToApiResult(enum.Enum):
    """Enum for representing the result of connecting to the FEDn API."""

    Assigned = 0
    ComputePackageMissing = 1
    UnAuthorized = 2
    UnMatchedConfig = 3
    IncorrectUrl = 4
    UnknownError = 5


class FednClient:
    """Client for interacting with the FEDn network."""

    def __init__(
        self, train_callback: Optional[callable] = None, validate_callback: Optional[callable] = None, predict_callback: Optional[callable] = None
    ) -> None:
        """Initialize the FednClient."""
        self.train_callback = train_callback
        self.validate_callback = validate_callback
        self.predict_callback = predict_callback
        self.forward_callback: Optional[callable] = None
        self.backward_callback: Optional[callable] = None

        self.grpc_handler: Optional[GrpcHandler] = None

        self._current_logging_context: Optional[LoggingContext] = None

    def set_train_callback(self, callback: callable) -> None:
        """Set the train callback."""
        self.train_callback = callback

    def set_validate_callback(self, callback: callable) -> None:
        """Set the validate callback."""
        self.validate_callback = callback

    def set_predict_callback(self, callback: callable) -> None:
        """Set the predict callback."""
        self.predict_callback = callback

    def set_forward_callback(self, callback: callable):
        self.forward_callback = callback

    def set_backward_callback(self, callback: callable):
        self.backward_callback = callback

    def connect_to_api(self, url: str, token: str, json: dict) -> Tuple[ConnectToApiResult, Any]:
        """Connect to the FEDn API."""
        url_endpoint = urljoin(url, "api/v1/clients/add")
        logger.info(f"Connecting to API endpoint: {url_endpoint}")

        try:
            response = requests.post(
                url=url_endpoint,
                json=json,
                allow_redirects=True,
                headers={"Authorization": f"{FEDN_AUTH_SCHEME} {token}"},
                timeout=REQUEST_TIMEOUT,
                verify=FEDN_CONNECT_API_SECURE,
            )

            if response.status_code == HTTP_STATUS_OK:
                logger.info("Connect to FEDn Api - Client assigned to controller")
                json_response = response.json()
                self.set_client_id(json_response["client_id"])
                self.set_name(json.get("name", json_response["client_id"]))
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

    def _send_heartbeats(self, client_name: str, client_id: str, update_frequency: float = 2.0) -> None:
        """Send heartbeats to the server."""
        self.grpc_handler.send_heartbeats(client_name=client_name, client_id=client_id, update_frequency=update_frequency)

    def _listen_to_task_stream(self, client_name: str, client_id: str) -> None:
        """Listen to the task stream."""
        self.grpc_handler.listen_to_task_stream(client_name=client_name, client_id=client_id, callback=self._task_stream_callback)

    def default_telemetry_loop(self, update_frequency: float = 5.0) -> None:
        """Send default telemetry data."""
        send_telemetry = True
        while send_telemetry:
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            try:
                success = self.log_telemetry(telemetry={"memory_usage": memory_usage, "cpu_usage": cpu_usage})
            except RetryException as e:
                logger.error(f"Sending telemetry failed: {e}")
            if not success:
                logger.error("Telemetry failed.")
                send_telemetry = False
            time.sleep(update_frequency)

    @contextmanager
    def logging_context(self, context: LoggingContext):
        """Set the logging context."""
        prev_context = self._current_logging_context
        self._current_logging_context = context
        try:
            yield
        finally:
            self._current_logging_context = prev_context

    def _task_stream_callback(self, request: fedn.TaskRequest) -> None:
        """Handle task stream callbacks."""
        if request.type == fedn.StatusType.MODEL_UPDATE:
            self.update_local_model(request)
        elif request.type == fedn.StatusType.MODEL_VALIDATION:
            self.validate_global_model(request)
        elif request.type == fedn.StatusType.MODEL_PREDICTION:
            self.predict_global_model(request)
        elif request.type == fedn.StatusType.FORWARD:
            self.forward_embeddings(request)
        elif request.type == fedn.StatusType.BACKWARD:
            self.backward_gradients(request)

    def update_local_model(self, request: fedn.TaskRequest) -> None:
        """Update the local model."""
        with self.logging_context(LoggingContext(request=request)):
            model_id = request.model_id
            model_update_id = str(uuid.uuid4())

            tic = time.time()
            in_model = self.get_model_from_combiner(model_id=model_id, client_id=self.client_id)

            if in_model is None:
                logger.error("Could not retrieve model from combiner. Aborting training request.")
                return

            fetch_model_time = time.time() - tic
            logger.info(f"FETCH_MODEL: {fetch_model_time}")

            if not self.train_callback:
                logger.error("No train callback set")
                return

            self.send_status(
                f"\t Starting processing of training request for model_id {model_id}",
                session_id=request.session_id,
                sender_name=self.name,
                log_level=fedn.LogLevel.INFO,
                type=fedn.StatusType.MODEL_UPDATE,
            )

            logger.info(f"Running train callback with model ID: {model_id}")
            client_settings = json.loads(request.data).get("client_settings", {})
            tic = time.time()
            try:
                out_model, meta = self.train_callback(in_model, client_settings)
            except Exception as e:
                logger.error(f"Train callback failed with expection: {e}")
                return
            meta["processing_time"] = time.time() - tic

            tic = time.time()
            self.send_model_to_combiner(model=out_model, model_id=model_update_id)
            meta["upload_model"] = time.time() - tic
            logger.info("UPLOAD_MODEL: {0}".format(meta["upload_model"]))

            meta["fetch_model"] = fetch_model_time
            meta["config"] = request.data

            update = self.grpc_handler.create_update_message(
                sender_name=self.name,
                model_id=model_id,
                model_update_id=model_update_id,
                receiver_name=request.sender.name,
                receiver_role=request.sender.role,
                meta=meta,
            )

            self.grpc_handler.send_model_update(update)

            self.send_status(
                "Model update completed.",
                log_level=fedn.LogLevel.AUDIT,
                type=fedn.StatusType.MODEL_UPDATE,
                request=update,
                session_id=request.session_id,
                sender_name=self.name,
            )

    def check_task_abort(self) -> None:
        # Check if the current task has been aborted
        # To be implemented
        """Raises an exception if the current task has been aborted. Does nothing for now."""
        pass

    def validate_global_model(self, request: fedn.TaskRequest) -> None:
        """Validate the global model."""
        with self.logging_context(LoggingContext(request=request)):
            model_id = request.model_id

            self.send_status(
                f"Processing validate request for model_id {model_id}",
                session_id=request.session_id,
                sender_name=self.name,
                log_level=fedn.LogLevel.INFO,
                type=fedn.StatusType.MODEL_VALIDATION,
            )

            in_model = self.get_model_from_combiner(model_id=model_id, client_id=self.client_id)

            if in_model is None:
                logger.error("Could not retrieve model from combiner. Aborting validation request.")
                return

            if not self.validate_callback:
                logger.error("No validate callback set")
                return

            logger.debug(f"Running validate callback with model ID: {model_id}")
            try:
                metrics = self.validate_callback(in_model)
            except Exception as e:
                logger.error(f"Validation callback failed with expection: {e}")
                return

            if metrics is not None:
                # Send validation

                validation = self.grpc_handler.create_validation_message(
                    sender_name=self.name,
                    sender_client_id=self.client_id,
                    receiver_name=request.sender.name,
                    receiver_role=request.sender.role,
                    model_id=request.model_id,
                    metrics=json.dumps(metrics),
                    correlation_id=request.correlation_id,
                    session_id=request.session_id,
                )

                result: bool = self.grpc_handler.send_model_validation(validation)

                if result:
                    self.send_status(
                        "Model validation completed.",
                        log_level=fedn.LogLevel.AUDIT,
                        type=fedn.StatusType.MODEL_VALIDATION,
                        request=validation,
                        session_id=request.session_id,
                        sender_name=self.name,
                    )
                else:
                    self.send_status(
                        f"Client {self.name} failed to complete model validation.",
                        log_level=fedn.LogLevel.WARNING,
                        request=request,
                        session_id=request.session_id,
                        sender_name=self.name,
                    )

    def predict_global_model(self, request: fedn.TaskRequest) -> None:
        """Predict using the global model."""
        with self.logging_context(LoggingContext(request=request)):
            model_id = request.model_id
            model = self.get_model_from_combiner(model_id=model_id, client_id=self.client_id)

            if model is None:
                logger.error("Could not retrieve model from combiner. Aborting prediction request.")
                return

            if not self.predict_callback:
                logger.error("No predict callback set")
                return

            logger.info(f"Running predict callback with model ID: {model_id}")
            try:
                prediction = self.predict_callback(model)
            except Exception as e:
                logger.error(f"Predict callback failed with expection: {e}")
                return

            prediction_message = self.grpc_handler.create_prediction_message(
                sender_name=self.name,
                receiver_name=request.sender.name,
                receiver_role=request.sender.role,
                model_id=request.model_id,
                prediction_output=json.dumps(prediction),
                correlation_id=request.correlation_id,
                session_id=request.session_id,
            )

            self.grpc_handler.send_model_prediction(prediction_message)

    def log_metric(self, metrics: dict, step: int = None, commit: bool = True) -> bool:
        """Log the metrics to the server.

        Args:
            metrics (dict): The metrics to log.
            step (int, optional): The step number.
            If provided the context step will be set to this value.
            If not provided, the step from the context will be used.
            commit (bool, optional): Whether or not to increment the step.  Defaults to True.

        Returns:
            bool: True if the metrics were logged successfully, False otherwise.

        """
        context = self._current_logging_context

        if context is None:
            logger.error("Missing context for logging metric.")
            return False

        if step is None:
            step = context.step
        else:
            context.step = step

        if commit:
            context.step += 1

        message = self.grpc_handler.create_metric_message(
            sender_name=self.name,
            sender_client_id=self.client_id,
            metrics=metrics,
            model_id=context.model_id,
            step=step,
            round_id=context.round_id,
            session_id=context.session_id,
        )

        return self.grpc_handler.send_model_metric(message)

    def forward_embeddings(self, request):
        """Forward pass for split learning gradient calculation or inference."""
        model_id = request.model_id
        is_sl_inference = json.loads(request.data).get("is_sl_inference", False)

        embedding_update_id = str(uuid.uuid4())

        if not self.forward_callback:
            logger.error("No forward callback set")
            return

        self.send_status(f"\t Starting processing of forward request for model_id {model_id}", session_id=request.session_id, sender_name=self.name)

        logger.info(f"Running forward callback with model ID: {model_id}")
        tic = time.time()
        out_embeddings, meta = self.forward_callback(self.client_id, is_sl_inference)
        meta["processing_time"] = time.time() - tic

        tic = time.time()
        self.send_model_to_combiner(model=out_embeddings, model_id=embedding_update_id)
        meta["upload_model"] = time.time() - tic

        meta["config"] = request.data

        update = self.grpc_handler.create_update_message(
            sender_name=self.name,
            model_id=model_id,
            model_update_id=embedding_update_id,
            receiver_name=request.sender.name,
            receiver_role=request.sender.role,
            meta=meta,
        )

        self.grpc_handler.send_model_update(update)

        self.send_status(
            "Forward pass completed.",
            log_level=fedn.LogLevel.AUDIT,
            type=fedn.StatusType.MODEL_UPDATE,
            request=update,
            session_id=request.session_id,
            sender_name=self.name,
        )

    def backward_gradients(self, request):
        """Split learning backward pass to update the local client models."""
        model_id = request.model_id

        try:
            tic = time.time()
            in_gradients = self.get_model_from_combiner(model_id=model_id, client_id=self.client_id)  # gets gradients

            if in_gradients is None:
                logger.error("Could not retrieve gradients from combiner. Aborting backward request.")
                return

            fetch_model_time = time.time() - tic

            if not self.backward_callback:
                logger.error("No backward callback set")
                return

            self.send_status(f"\t Starting processing of backward request for gradient_id {model_id}", session_id=request.session_id, sender_name=self.name)

            logger.info(f"Running backward callback with gradient ID: {model_id}")
            tic = time.time()
            meta = self.backward_callback(in_gradients, self.client_id)
            meta["processing_time"] = time.time() - tic

            meta["fetch_model"] = fetch_model_time
            meta["config"] = request.data
            meta["status"] = "success"

            logger.info("Creating and sending backward completion to combiner.")

            completion = self.grpc_handler.create_backward_completion_message(
                sender_name=self.name,
                receiver_name=request.sender.name,
                receiver_role=request.sender.role,
                gradient_id=model_id,
                session_id=request.session_id,
                meta=meta,
            )

            self.grpc_handler.send_backward_completion(completion)

            self.send_status(
                "Backward pass completed. Status: finished_backward",
                log_level=fedn.LogLevel.AUDIT,
                type=fedn.StatusType.BACKWARD,
                session_id=request.session_id,
                sender_name=self.name,
            )
        except Exception as e:
            logger.error(f"Error in backward pass: {str(e)}")

    def log_attributes(self, attributes: dict) -> bool:
        """Log the attributes to the server.

        Args:
            attributes (dict): The attributes to log.

        Returns:
            bool: True if the attributes were logged successfully, False otherwise.

        """
        message = fedn.AttributeMessage()
        message.sender.name = self.name
        message.sender.client_id = self.client_id
        message.sender.role = fedn.Role.CLIENT
        message.timestamp.GetCurrentTime()

        for key, value in attributes.items():
            message.attributes.add(key=key, value=value)

        return self.grpc_handler.send_attributes(message)

    def log_telemetry(self, telemetry: dict) -> bool:
        """Log the telemetry data to the server.

        Args:
            telemetry (dict): The telemetry data to log.

        Returns:
            bool: True if the telemetry data was logged successfully, False otherwise.

        """
        message = fedn.TelemetryMessage()
        message.sender.name = self.name
        message.sender.client_id = self.client_id
        message.sender.role = fedn.Role.CLIENT
        message.timestamp.GetCurrentTime()

        for key, value in telemetry.items():
            message.telemetries.add(key=key, value=value)

        return self.grpc_handler.send_telemetry(message)

    def set_name(self, name: str) -> None:
        """Set the client name."""
        logger.info(f"Setting client name to: {name}")
        self.name = name

    def set_client_id(self, client_id: str) -> None:
        """Set the client ID."""
        logger.info(f"Setting client ID to: {client_id}")
        self.client_id = client_id

    def run(self, with_telemetry=True, with_heartbeat=True) -> None:
        """Run the client."""
        if with_heartbeat:
            threading.Thread(target=self._send_heartbeats, args=(self.name, self.client_id), daemon=True).start()
        if with_telemetry:
            threading.Thread(target=self.default_telemetry_loop, daemon=True).start()
        try:
            self._listen_to_task_stream(client_name=self.name, client_id=self.client_id)
        except KeyboardInterrupt:
            logger.info("Client stopped by user.")

    def get_model_from_combiner(self, model_id: str, client_id: str, timeout: int = 20) -> BytesIO:
        """Get the model from the combiner."""
        return self.grpc_handler.get_model_from_combiner(model_id=model_id, client_id=client_id, timeout=timeout)

    def send_model_to_combiner(self, model: BytesIO, model_id: str) -> None:
        """Send the model to the combiner."""
        self.grpc_handler.send_model_to_combiner(model, model_id)

    def send_status(
        self,
        msg: str,
        log_level: fedn.LogLevel = fedn.LogLevel.INFO,
        type: Optional[str] = None,
        request: Optional[Union[fedn.ModelUpdate, fedn.ModelValidation, fedn.TaskRequest]] = None,
        session_id: Optional[str] = None,
        sender_name: Optional[str] = None,
    ) -> None:
        """Send the status."""
        self.grpc_handler.send_status(msg, log_level, type, request, session_id, sender_name)
