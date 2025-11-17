"""FednClient class for interacting with the FEDn network."""

import enum
import json
import signal
import threading
import time
import uuid
from contextlib import contextmanager
from io import BytesIO
from typing import Any, Callable, Dict, Optional, Tuple

from scaleoututil.utils.dist import get_version
from scaleoututil.utils.url import assemble_endpoint_url
import psutil
import requests

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
from scaleoututil.config import (
    SCALEOUT_AUTH_SCHEME,
    SCALEOUT_CONNECT_API_SECURE,
    SCALEOUT_CLIENT_STATUS_REPORTING,
    SCALEOUT_CLIENT_SEND_TELEMETRY,
    SCALEOUT_GRACEFUL_CLIENT_CONNECTION,
    SCALEOUT_CHECK_COMPATIBILITY,
    SCALEOUT_CLIENT_TASK_POLLING_INTERVAL,
)
from scaleoututil.logging import FednLogger
from scaleout.client.grpc_handler import GrpcConnectionOptions, GrpcHandler, RetryException
from scaleoututil.utils.http_status_codes import (
    HTTP_STATUS_BAD_REQUEST,
    HTTP_STATUS_NOT_ACCEPTABLE,
    HTTP_STATUS_NOT_FOUND,
    HTTP_STATUS_OK,
    HTTP_STATUS_PACKAGE_MISSING,
    HTTP_STATUS_UNAUTHORIZED,
)
from scaleout.client.logging_context import LoggingContext
from scaleout.client.task_receiver import TaskReceiver, UnknownTaskType
from scaleoututil.grpc.tasktype import TaskType

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


class GracefulExitException(Exception):
    pass


class FednClient:
    """Client for interacting with the FEDn network."""

    def __init__(
        self, train_callback: Optional[callable] = None, validate_callback: Optional[callable] = None, predict_callback: Optional[callable] = None
    ) -> None:
        """Initialize the FednClient."""
        self.name: str = None
        self.client_id: str = None

        self.train_callback = train_callback
        self.validate_callback = validate_callback
        self.predict_callback = predict_callback
        self.forward_callback: Optional[callable] = None
        self.backward_callback: Optional[callable] = None

        self.grpc_handler: Optional[GrpcHandler] = None

        self._current_logging_context = threading.local()

        self.task_receiver = TaskReceiver(self, self._run_task_callback, polling_interval=SCALEOUT_CLIENT_TASK_POLLING_INTERVAL)
        self.registered_callbacks: Dict[str, Callable[[scaleout_msg.TaskRequest], Dict]] = {}

    @property
    def current_logging_context(self) -> Optional[LoggingContext]:
        """Get the current logging context for the running thread."""
        return getattr(self._current_logging_context, "value", None)

    @current_logging_context.setter
    def current_logging_context(self, context: LoggingContext) -> None:
        """Set the current logging context for the running thread."""
        self._current_logging_context.value = context

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

    def set_custom_callback(self, callback_name: str, callback: Callable[[scaleout_msg.TaskRequest], Dict]) -> None:
        """Set a custom task callback."""
        if not callback_name.startswith("Custom_"):
            callback_name = "Custom_" + callback_name
        self.registered_callbacks[callback_name] = callback
        FednLogger().info(f"Registered custom callback: {callback_name}")

    def remove_custom_callback(self, callback_name: str) -> None:
        """Remove a custom task callback."""
        if not callback_name.startswith("Custom_"):
            callback_name = "Custom_" + callback_name
        if callback_name in self.registered_callbacks:
            del self.registered_callbacks[callback_name]
            FednLogger().info(f"Removed custom callback: {callback_name}")
        else:
            FednLogger().warning(f"Custom callback {callback_name} not found")

    def connect_to_api(self, url: str, token: str, json: dict) -> Tuple[ConnectToApiResult, Any]:
        """Connect to the FEDn API."""
        url_endpoint = assemble_endpoint_url(url, "api/v1/clients/add")
        FednLogger().info(f"Connecting to API endpoint: {url_endpoint}")
        if SCALEOUT_CHECK_COMPATIBILITY:
            json["client_version"] = get_version()

        try:
            response = requests.post(
                url=url_endpoint,
                json=json,
                allow_redirects=True,
                headers={"Authorization": f"{SCALEOUT_AUTH_SCHEME} {token}"},
                timeout=REQUEST_TIMEOUT,
                verify=SCALEOUT_CONNECT_API_SECURE,
            )

            if response.status_code == HTTP_STATUS_OK:
                FednLogger().info("Connect to FEDn Api - Client assigned to controller")
                json_response = response.json()
                self.set_client_id(json_response["client_id"])
                self.set_name(json.get("name", json_response["client_id"]))
                combiner_config = GrpcConnectionOptions.from_dict(json_response)
                return ConnectToApiResult.Assigned, combiner_config

            if response.status_code == HTTP_STATUS_PACKAGE_MISSING:
                json_response = response.json()
                FednLogger().info("Connect to FEDn Api - Remote compute package missing.")
                return ConnectToApiResult.ComputePackageMissing, json_response

            if response.status_code == HTTP_STATUS_UNAUTHORIZED:
                FednLogger().error("Connect to FEDn Api - Unauthorized")
                return ConnectToApiResult.UnAuthorized, "Unauthorized"

            if response.status_code == HTTP_STATUS_BAD_REQUEST:
                try:
                    json_response = response.json()
                except Exception:
                    json_response = {}
                msg = json_response.get("message", "Unknown error")
                FednLogger().error(f"Connect to FEDn Api - {msg}")
                return ConnectToApiResult.UnMatchedConfig, msg

            if response.status_code == HTTP_STATUS_NOT_ACCEPTABLE:
                try:
                    json_response = response.json()
                except Exception:
                    json_response = {}
                msg = json_response.get("message", "Unknown error")
                FednLogger().error(f"Connect to FEDn Api - {msg}")
                return ConnectToApiResult.UnMatchedConfig, msg

            if response.status_code == HTTP_STATUS_NOT_FOUND:
                FednLogger().error("Connect to FEDn Api - Incorrect URL")
                return ConnectToApiResult.IncorrectUrl, "Incorrect URL"

        except Exception as e:
            FednLogger().error(f"Connect to FEDn Api - Error occurred: {str(e)}")
            return ConnectToApiResult.UnknownError, str(e)

    def init_grpchandler(self, config: GrpcConnectionOptions, client_id: str, token: str) -> bool:
        """Initialize the GRPC handler."""
        try:
            if config.fqdn and len(config.fqdn) > 0:
                host = config.fqdn
                port = 443
            else:
                host = config.host
                port = config.port

            self.grpc_handler = GrpcHandler(self, host=host, port=port, token=token)

            if SCALEOUT_CHECK_COMPATIBILITY:
                success, server_version, msg = self.grpc_handler.check_version_compatibility()
                if not success:
                    FednLogger().error(f"Client version: {get_version()} compatibility check failed with Server version: {server_version}. {msg}")
                    return False

            FednLogger().info("Successfully initialized GRPC connection")
            return True
        except Exception as e:
            FednLogger().error(f"Could not initialize GRPC connection: {e}")
            return False

    def _send_heartbeats(self, client_name: str, client_id: str, update_frequency: float = 2.0) -> None:
        """Send heartbeats to the server."""
        self.grpc_handler.send_heartbeats(client_name=client_name, client_id=client_id, update_frequency=update_frequency)

    def _listen_to_task_stream(self, client_id: str) -> None:
        """Listen to the task stream."""
        self.grpc_handler.listen_to_task_stream(client_id=client_id, callback=self._task_stream_callback)

    def default_telemetry_loop(self, update_frequency: float = 5.0) -> None:
        """Send default telemetry data."""
        send_telemetry = True
        while send_telemetry:
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            try:
                success = self.log_telemetry(telemetry={"memory_usage": memory_usage, "cpu_usage": cpu_usage})
            except RetryException as e:
                FednLogger().error(f"Sending telemetry failed: {e}")
            if not success:
                FednLogger().error("Telemetry failed.")
                send_telemetry = False
            time.sleep(update_frequency)

    @contextmanager
    def logging_context(self, context: LoggingContext):
        """Set the logging context."""
        prev_context = self.current_logging_context
        self.current_logging_context = context
        try:
            yield
        finally:
            self.current_logging_context = prev_context

    def _task_stream_callback(self, request: scaleout_msg.TaskRequest) -> None:
        """Handle task stream callbacks."""
        if request.type == TaskType.ModelUpdate.value:
            self.update_local_model(request)
        elif request.type == TaskType.Validation.value:
            self.validate_global_model(request)
        elif request.type == TaskType.Prediction.value:
            self.predict_global_model(request)
        elif request.type == TaskType.Forward.value:
            self.forward_embeddings(request)
        elif request.type == TaskType.Backward.value:
            self.backward_gradients(request)
        return {}

    def _run_task_callback(self, request: scaleout_msg.TaskRequest) -> None:
        if request.type in (t.value for t in TaskType):
            return self._task_stream_callback(request)
        elif TaskType.is_custom_task(request.type):
            return self._handle_custom_task(request)
        else:
            FednLogger().error(f"Invalid task type: {request.type}")
            raise Exception(f"Invalid task type: {request.type}")

    def _handle_custom_task(self, request: scaleout_msg.TaskRequest) -> None:
        if request.type in self.registered_callbacks:
            with self.logging_context(LoggingContext(request=request)):
                params = json.loads(request.data) if request.data else {}
                result = self.registered_callbacks[request.type](params)
                return result
        else:
            FednLogger().warning(f"Unknown task type: {request.type}")
            raise UnknownTaskType(f"Unknown task type: {request.type}")

    def update_local_model(self, request: scaleout_msg.TaskRequest) -> None:
        """Update the local model."""
        with self.logging_context(LoggingContext(request=request)):
            model_id = request.model_id
            model_update_id = str(uuid.uuid4())

            tic = time.time()
            in_model = self.get_model_from_combiner(model_id=model_id)

            if in_model is None:
                FednLogger().error("Could not retrieve model from combiner. Aborting training request.")
                return

            fetch_model_time = time.time() - tic
            FednLogger().info(f"FETCH_MODEL: {fetch_model_time}")

            if not self.train_callback:
                FednLogger().error("No train callback set")
                return

            if SCALEOUT_CLIENT_STATUS_REPORTING:
                self.send_status(
                    f"\t Starting processing of training request for model_id {model_id}",
                    log_level=scaleout_msg.LogLevel.INFO,
                    type="MODEL_UPDATE",
                )

            FednLogger().info(f"Running train callback with model ID: {model_id}")
            client_settings = json.loads(request.data).get("client_settings", {})
            tic = time.time()
            try:
                out_model, meta = self.train_callback(in_model, client_settings)
            except Exception as e:
                FednLogger().error(f"Train callback failed with expection: {e}")
                return
            meta["processing_time"] = time.time() - tic

            tic = time.time()
            self.send_model_to_combiner(model=out_model, model_id=model_update_id)
            meta["upload_model"] = time.time() - tic
            FednLogger().info("UPLOAD_MODEL: {0}".format(meta["upload_model"]))

            meta["fetch_model"] = fetch_model_time
            meta["config"] = request.data

            update = self.create_update_message(model_id=model_id, model_update_id=model_update_id, meta=meta, request=request)

            self.grpc_handler.send_model_update(update)

            if SCALEOUT_CLIENT_STATUS_REPORTING:
                self.send_status(
                    "Model update completed.",
                    log_level=scaleout_msg.LogLevel.AUDIT,
                    type="MODEL_UPDATE",
                )

    def validate_global_model(self, request: scaleout_msg.TaskRequest) -> None:
        """Validate the global model."""
        with self.logging_context(LoggingContext(request=request)):
            model_id = request.model_id

            if SCALEOUT_CLIENT_STATUS_REPORTING:
                self.send_status(
                    f"Processing validate request for model_id {model_id}",
                    log_level=scaleout_msg.LogLevel.INFO,
                    type="MODEL_VALIDATION",
                )

            in_model = self.get_model_from_combiner(model_id=model_id)

            if in_model is None:
                FednLogger().error("Could not retrieve model from combiner. Aborting validation request.")
                return

            if not self.validate_callback:
                FednLogger().error("No validate callback set")
                return

            FednLogger().debug(f"Running validate callback with model ID: {model_id}")
            try:
                metrics = self.validate_callback(in_model)
            except Exception as e:
                FednLogger().error(f"Validation callback failed with expection: {e}")
                return

            if metrics is not None:
                # Send validation

                validation = self.grpc_handler.create_validation_message(
                    model_id=request.model_id,
                    metrics=json.dumps(metrics),
                    correlation_id=request.correlation_id,
                    session_id=request.session_id,
                )

                result: bool = self.grpc_handler.send_model_validation(validation)

                if result and SCALEOUT_CLIENT_STATUS_REPORTING:
                    self.send_status(
                        "Model validation completed.",
                        log_level=scaleout_msg.LogLevel.AUDIT,
                        type="MODEL_VALIDATION",
                    )
                elif SCALEOUT_CLIENT_STATUS_REPORTING:
                    self.send_status(
                        f"Client {self.client_id} failed to complete model validation.",
                        log_level=scaleout_msg.LogLevel.WARNING,
                        type="MODEL_VALIDATION",
                    )

    def predict_global_model(self, request: scaleout_msg.TaskRequest) -> None:
        """Predict using the global model."""
        with self.logging_context(LoggingContext(request=request)):
            model_id = request.model_id
            model = self.get_model_from_combiner(model_id=model_id)

            if model is None:
                FednLogger().error("Could not retrieve model from combiner. Aborting prediction request.")
                return

            if not self.predict_callback:
                FednLogger().error("No predict callback set")
                return

            FednLogger().info(f"Running predict callback with model ID: {model_id}")
            try:
                prediction = self.predict_callback(model)
            except Exception as e:
                FednLogger().error(f"Predict callback failed with expection: {e}")
                return

            prediction_message = self.grpc_handler.create_prediction_message(
                model_id=request.model_id,
                prediction_output=json.dumps(prediction),
                correlation_id=request.correlation_id,
                session_id=request.session_id,
            )

            self.grpc_handler.send_model_prediction(prediction_message)

    def log_metric(self, metrics: dict, step: int = None, commit: bool = True, check_task_abort=True, context: LoggingContext = None) -> bool:
        """Log the metrics to the server.

        Args:
            metrics (dict): The metrics to log.
            step (int, optional): The step number.
            If provided the context step will be set to this value.
            If not provided, the step from the context will be used.
            commit (bool, optional): Whether or not to increment the step.  Defaults to True.
            check_task_abort (bool, optional): Whether or not to check for task abort. Defaults to True.
            context (LoggingContext, optional): The logging context to use. Defaults to None, which uses the current context.

        Returns:
            bool: True if the metrics were logged successfully, False otherwise.

        """
        context = context or self.current_logging_context

        if context is None:
            FednLogger().error("Missing context for logging metric.")
            return False

        if step is None:
            step = context.step
        else:
            context.step = step

        if commit:
            context.step += 1

        message = self.grpc_handler.create_metric_message(
            metrics=metrics,
            model_id=context.model_id,
            step=step,
            round_id=context.round_id,
            session_id=context.session_id,
        )

        success = self.grpc_handler.send_model_metric(message)
        if check_task_abort:
            self.task_receiver.check_abort()
        return success

    def forward_embeddings(self, request):
        """Forward pass for split learning gradient calculation or inference."""
        model_id = request.model_id
        is_sl_inference = json.loads(request.data).get("is_sl_inference", False)

        embedding_update_id = str(uuid.uuid4())

        if not self.forward_callback:
            FednLogger().error("No forward callback set")
            return

        if SCALEOUT_CLIENT_STATUS_REPORTING:
            self.send_status(f"\t Starting processing of forward request for model_id {model_id}", type="SPLITLEARNING_FORWARD")

        FednLogger().info(f"Running forward callback with model ID: {model_id}")
        tic = time.time()
        out_embeddings, meta = self.forward_callback(self.client_id, is_sl_inference)
        meta["processing_time"] = time.time() - tic

        tic = time.time()
        self.send_model_to_combiner(model=out_embeddings, model_id=embedding_update_id)
        meta["upload_model"] = time.time() - tic

        meta["config"] = request.data

        update = self.create_update_message(model_id=model_id, model_update_id=embedding_update_id, meta=meta, request=request)

        self.grpc_handler.send_model_update(update)

        if SCALEOUT_CLIENT_STATUS_REPORTING:
            self.send_status(
                "Forward pass completed.",
                log_level=scaleout_msg.LogLevel.AUDIT,
                type="SPLITLEARNING_FORWARD",
            )

    def backward_gradients(self, request: scaleout_msg.TaskRequest):
        """Split learning backward pass to update the local client models."""
        model_id = request.model_id

        try:
            tic = time.time()
            in_gradients = self.get_model_from_combiner(model_id=model_id)  # gets gradients

            if in_gradients is None:
                FednLogger().error("Could not retrieve gradients from combiner. Aborting backward request.")
                return {}

            fetch_model_time = time.time() - tic

            if not self.backward_callback:
                FednLogger().error("No backward callback set")
                return {}

            if SCALEOUT_CLIENT_STATUS_REPORTING:
                self.send_status(f"\t Starting processing of backward request for gradient_id {model_id}", type="SPLITLEARNING_BACKWARD")

            FednLogger().info(f"Running backward callback with gradient ID: {model_id}")
            tic = time.time()
            meta = self.backward_callback(in_gradients, self.client_id)
            meta["processing_time"] = time.time() - tic

            meta["fetch_model"] = fetch_model_time
            meta["status"] = "success"

            FednLogger().info("Creating and sending backward completion to combiner.")

            completion = self.create_backward_completion_message(gradient_id=model_id, request=request)
            self.grpc_handler.send_backward_completion(completion)

            if SCALEOUT_CLIENT_STATUS_REPORTING:
                self.send_status(
                    "Backward pass completed. Status: finished_backward",
                    log_level=scaleout_msg.LogLevel.AUDIT,
                    type="SPLITLEARNING_BACKWARD",
                )
            return meta
        except Exception as e:
            FednLogger().error(f"Error in backward pass: {str(e)}")

    def create_backward_completion_message(self, gradient_id: str, request: scaleout_msg.TaskRequest):
        """Create a backward completion message."""
        return self.grpc_handler.create_backward_completion_message(gradient_id=gradient_id, session_id=request.session_id)

    def log_attributes(self, attributes: dict, check_task_abort: bool = True) -> bool:
        """Log the attributes to the server.

        Args:
            attributes (dict): The attributes to log.
            check_task_abort (bool, optional): Whether or not to check for task abort. Defaults to True.

        Returns:
            bool: True if the attributes were logged successfully, False otherwise.

        """
        message = scaleout_msg.AttributeMessage()
        message.client_id = self.client_id
        message.timestamp.GetCurrentTime()

        for key, value in attributes.items():
            message.attributes.add(key=key, value=value)

        success = self.grpc_handler.send_attributes(message)
        if check_task_abort:
            self.task_receiver.check_abort()
        return success

    def log_telemetry(
        self,
        telemetry: dict,
        check_task_abort: bool = True,
    ) -> bool:
        """Log the telemetry data to the server.

        Args:
            telemetry (dict): The telemetry data to log.
            check_task_abort (bool, optional): Whether or not to check for task abort. Defaults to True.

        Returns:
            bool: True if the telemetry data was logged successfully, False otherwise.

        """
        message = scaleout_msg.TelemetryMessage()
        message.client_id = self.client_id
        message.timestamp.GetCurrentTime()

        for key, value in telemetry.items():
            message.telemetries.add(key=key, value=value)

        success = self.grpc_handler.send_telemetry(message)
        if check_task_abort:
            self.task_receiver.check_abort()
        return success

    def check_task_abort(self) -> None:
        """Check if the ongoing task has been aborted.

        This function should be called periodically from the task callback to ensure
        that the task can be interrupted if needed.
        If called from a thread that do not run the task, this function is a no-op.

        Raises:
            StoppedException: If the task was aborted.

        """
        self.task_receiver.check_abort()

    def create_update_message(self, model_id: str, model_update_id: str, meta: dict, request: scaleout_msg.TaskRequest) -> scaleout_msg.ModelUpdate:
        """Create an update message."""
        return self.grpc_handler.create_update_message(
            model_id=model_id,
            model_update_id=model_update_id,
            correlation_id=request.correlation_id,
            round_id=request.round_id,
            session_id=request.session_id,
            meta=meta,
        )

    def create_prediction_message(self, prediction: dict, request: scaleout_msg.TaskRequest) -> scaleout_msg.ModelPrediction:
        """Create a prediction message."""
        return self.grpc_handler.create_prediction_message(
            model_id=request.model_id,
            prediction_output=json.dumps(prediction),
            correlation_id=request.correlation_id,
            session_id=request.session_id,
        )

    def set_name(self, name: str) -> None:
        """Set the client name."""
        FednLogger().info(f"Setting client name to: {name}")
        self.name = name

    def set_client_id(self, client_id: str) -> None:
        """Set the client ID."""
        FednLogger().info(f"Setting client ID to: {client_id}")
        self.client_id = client_id

    def run(self, with_heartbeat=False, with_polling=True) -> None:
        """Run the client."""

        # Handle SIGTERM for graceful shutdown
        def _handle_sigterm(signum, frame):
            raise GracefulExitException()

        signal.signal(signal.SIGTERM, _handle_sigterm)

        if with_heartbeat:
            threading.Thread(target=self._send_heartbeats, args=(self.name, self.client_id), daemon=True).start()
        if SCALEOUT_CLIENT_SEND_TELEMETRY:
            threading.Thread(target=self.default_telemetry_loop, daemon=True).start()

        try:
            if with_polling:
                self._run_polling_client()
            else:
                self._listen_to_task_stream(client_id=self.client_id)
        except KeyboardInterrupt:
            FednLogger().info("Client stopped by user.")
        except GracefulExitException:
            FednLogger().info("Client stopping gracefully.")

    def _run_polling_client(self) -> None:
        self.task_receiver.start()
        FednLogger().info("Task receiver started.")
        if SCALEOUT_GRACEFUL_CLIENT_CONNECTION:
            self.grpc_handler.connect()
        while True:
            try:
                FednLogger().info("Client is running. Press Ctrl+C to stop.")
                self.task_receiver.wait_on_manager_thread()
                FednLogger().info("Task manager thread has exited. Stopping client.")
                break
            except GracefulExitException:
                FednLogger().info("SIGTERM received, shutting down gracefully...")
                if self.task_receiver.current_task is None:
                    FednLogger().info("No ongoing task to abort. Exiting...")
                    break
                self.task_receiver.abort_current_task()
                break
            except KeyboardInterrupt:
                FednLogger().info("KeyboardInterrupt received, aborting current task...")
                if self.task_receiver.current_task is None:
                    FednLogger().info("No ongoing task to abort. Exiting client.")
                    break
                self.task_receiver.abort_current_task()
                FednLogger().info("To completely stop the client, press Ctrl+C again within 5 seconds...")
            try:
                time.sleep(5)
            except KeyboardInterrupt:
                FednLogger().info("Second KeyboardInterrupt received, stopping client immediately...")
                break
        if SCALEOUT_GRACEFUL_CLIENT_CONNECTION:
            self.grpc_handler.disconnect()

    def get_model_from_combiner(self, model_id: str) -> BytesIO:
        """Get the model from the combiner."""
        return self.grpc_handler.get_model_from_combiner(model_id=model_id)

    def send_model_to_combiner(self, model: BytesIO, model_id: str) -> None:
        """Send the model to the combiner."""
        self.grpc_handler.send_model_to_combiner(model, model_id)

    def send_status(
        self,
        msg: str,
        log_level: scaleout_msg.LogLevel = scaleout_msg.LogLevel.INFO,
        type: Optional[str] = None,
    ) -> None:
        """Send the status."""
        self.grpc_handler.send_status(msg, log_level, type)
