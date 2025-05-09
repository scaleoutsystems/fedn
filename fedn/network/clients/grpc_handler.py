"""GrpcHandler class for handling GRPC connections and operations."""

import json
import os
import random
import time
from datetime import datetime, timezone
from functools import wraps
from io import BytesIO
from typing import Any, Callable, Optional, Union

import grpc
from google.protobuf.json_format import MessageToJson

import fedn.network.grpc.fedn_pb2 as fedn
import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.common.config import FEDN_AUTH_SCHEME
from fedn.common.log_config import logger
from fedn.network.combiner.modelservice import upload_request_generator

# Keepalive settings: these help keep the connection open for long-lived clients

KEEPALIVE_TIME_MS = 5 * 1000  # send keepalive ping every 5 second
# wait 30 seconds for keepalive ping ack before considering connection dead
KEEPALIVE_TIMEOUT_MS = 30 * 1000
# allow keepalive pings even when there are no RPCs
KEEPALIVE_PERMIT_WITHOUT_CALLS = True

GRPC_OPTIONS = [
    ("grpc.keepalive_time_ms", KEEPALIVE_TIME_MS),
    ("grpc.keepalive_timeout_ms", KEEPALIVE_TIMEOUT_MS),
    ("grpc.keepalive_permit_without_calls", KEEPALIVE_PERMIT_WITHOUT_CALLS),
]

GRPC_SECURE_PORT = 443


class GrpcAuth(grpc.AuthMetadataPlugin):
    """GRPC authentication plugin."""

    def __init__(self, key: str) -> None:
        """Initialize GrpcAuth with a key."""
        self._key = key

    def __call__(self, context: grpc.AuthMetadataContext, callback: grpc.AuthMetadataPluginCallback) -> None:
        """Add authorization metadata to the GRPC call."""
        callback((("authorization", f"{FEDN_AUTH_SCHEME} {self._key}"),), None)


class RetryException(Exception):
    pass


def grpc_retry(
    max_retries: int = 3,
    retry_interval: float = 5,
    backoff: float = 2,
) -> Callable:
    """GRPC retry decorator.


    :param max_retries: The maximum number of retries. -1 means infinite retries.
    :type max_retries: int
    :param retry_interval: The interval between retries in seconds.
    :type retry_interval: float
    :return: The decorated function.
    :rtype: Callable
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: "GrpcHandler", *args, **kwargs):
            """Wrapper function for retrying GRPC calls."""
            retries = 0
            last_try = time.time()
            backoff_factor = 1.0
            while max_retries > retries or max_retries == -1:
                retries += 1
                backoff_factor *= backoff

                # Reset backoff factor if the last try was more than 16 times the retry interval ago
                # This is to prevent the backoff factor from growing too large
                # if the server is down for a long time and then comes back up
                this_try = time.time()
                if this_try - last_try > 16 * retry_interval:
                    backoff_factor = 1.0
                last_try = this_try

                try:
                    return func(self, *args, **kwargs)
                except grpc.RpcError as e:
                    status_code = e.code()
                    if status_code == grpc.StatusCode.UNAVAILABLE:
                        logger.warning(f"GRPC ({func.__name__}): Server unavailable. Retrying in approx {retry_interval * backoff_factor} seconds.")
                        logger.debug(f"GRPC ({func.__name__}): Error details: {e.details()}")
                        self._reconnect()
                        time.sleep(retry_interval * backoff_factor + random.uniform(-0.5, 0.5))
                        continue
                    if status_code == grpc.StatusCode.CANCELLED:
                        logger.warning(f"GRPC ({func.__name__}): Connection cancelled. Retrying in approx {retry_interval * backoff_factor} seconds.")
                        logger.debug(f"GRPC ({func.__name__}): Error details: {e.details()}")
                        time.sleep(retry_interval * backoff_factor + random.uniform(-0.5, 0.5))
                        continue
                    if status_code == grpc.StatusCode.UNKNOWN:
                        details = e.details()
                        if details == "Stream removed":
                            logger.warning(f"GRPC ({func.__name__}): Stream removed. Retrying in approx {retry_interval * backoff_factor} seconds.")
                            self._reconnect()
                            time.sleep(retry_interval * backoff_factor + random.uniform(-0.5, 0.5))
                            continue
                        raise e
                    raise e
                except Exception as e:
                    logger.warning(f"GRPC ({func.__name__}): An unknown error occurred: {e}.")
                    if isinstance(e, ValueError):
                        logger.warning(f"GRPC ({func.__name__}): Retrying in approx {retry_interval * backoff_factor} seconds.")
                        self._reconnect()
                        time.sleep(retry_interval * backoff_factor + random.uniform(-0.5, 0.5))
                        continue
                    raise e

            logger.error(f"GRPC ({func.__name__}): Max retries exceeded.")
            raise RetryException("Max retries exceeded")

        return wrapper

    return decorator


class GrpcHandler:
    """Handler for GRPC connections and operations."""

    def __init__(self, host: str, port: int, name: str, token: str, combiner_name: str) -> None:
        """Initialize the GrpcHandler."""
        self.metadata = [
            ("client", name),
            ("grpc-server", combiner_name),
        ]
        self.host = host
        self.port = port
        self.token = token

        self._init_channel(host, port, token)
        self._init_stubs()

    def _init_stubs(self) -> None:
        """Initialize GRPC stubs."""
        self.connectorStub = rpc.ConnectorStub(self.channel)
        self.combinerStub = rpc.CombinerStub(self.channel)
        self.modelStub = rpc.ModelServiceStub(self.channel)

    def _init_channel(self, host: str, port: int, token: str) -> None:
        """Initialize the GRPC channel."""
        if port == GRPC_SECURE_PORT:
            self._init_secure_channel(host, port, token)
        else:
            self._init_insecure_channel(host, port)

    def _init_secure_channel(self, host: str, port: int, token: str) -> None:
        """Initialize a secure GRPC channel."""
        url = f"{host}:{port}"
        logger.info(f"Connecting (GRPC) to {url}")

        if os.getenv("FEDN_GRPC_ROOT_CERT_PATH"):
            logger.info("Using root certificate from environment variable for GRPC channel.")
            with open(os.environ["FEDN_GRPC_ROOT_CERT_PATH"], "rb") as f:
                credentials = grpc.ssl_channel_credentials(f.read())
            self.channel = grpc.secure_channel(
                f"{host}:{port}",
                credentials,
                options=GRPC_OPTIONS,
            )
            return

        credentials = grpc.ssl_channel_credentials()
        auth_creds = grpc.metadata_call_credentials(GrpcAuth(token))
        self.channel = grpc.secure_channel(
            f"{host}:{port}",
            grpc.composite_channel_credentials(credentials, auth_creds),
            options=GRPC_OPTIONS,
        )

    def _init_insecure_channel(self, host: str, port: int) -> None:
        """Initialize an insecure GRPC channel."""
        url = f"{host}:{port}"
        logger.info(f"Connecting (GRPC) to {url}")
        self.channel = grpc.insecure_channel(
            url,
            options=GRPC_OPTIONS,
        )

    def heartbeat(self, client_name: str, client_id: str, memory_utilisation: float = None, cpu_utilisation: float = None) -> fedn.Response:
        """Send a heartbeat to the combiner.

        :return: Response from the combiner.
        :rtype: fedn.Response
        """
        heartbeat = fedn.Heartbeat(sender=fedn.Client(name=client_name, role=fedn.CLIENT, client_id=client_id))

        response = self.connectorStub.SendHeartbeat(heartbeat, metadata=self.metadata)

        return response

    @grpc_retry(max_retries=-1, retry_interval=5)
    def send_heartbeats(self, client_name: str, client_id: str, update_frequency: float = 2.0) -> None:
        """Send heartbeats to the combiner at regular intervals."""
        send_heartbeat = True
        while send_heartbeat:
            response = self.heartbeat(client_name, client_id)
            time.sleep(update_frequency)
            if isinstance(response, fedn.Response):
                pass
            else:
                logger.error("Heartbeat failed.")
                send_heartbeat = False

    @grpc_retry(max_retries=-1, retry_interval=5)
    def listen_to_task_stream(self, client_name: str, client_id: str, callback: Callable[[Any], None]) -> None:
        """Subscribe to the model update request stream."""
        r = fedn.ClientAvailableMessage()
        r.sender.name = client_name
        r.sender.role = fedn.CLIENT
        r.sender.client_id = client_id

        logger.info("Listening to task stream.")
        for request in self.combinerStub.TaskStream(r, metadata=self.metadata):
            if request.sender.role == fedn.COMBINER:
                self.send_status(
                    "Received request from combiner.",
                    log_level=fedn.LogLevel.AUDIT,
                    type=request.type,
                    request=request,
                    session_id=request.session_id,
                    sender_name=client_name,
                )

                logger.info(f"Received task request of type {request.type} for model_id {request.model_id}")
                callback(request)

    @grpc_retry(max_retries=5, retry_interval=5)
    def send_status(
        self,
        msg: str,
        log_level: fedn.LogLevel = fedn.LogLevel.INFO,
        type: Optional[str] = None,
        request: Optional[Union[fedn.ModelUpdate, fedn.ModelValidation, fedn.TaskRequest]] = None,
        session_id: Optional[str] = None,
        sender_name: Optional[str] = None,
    ) -> None:
        """Send status message.

        :param msg: The message to send.
        :type msg: str
        :param log_level: The log level of the message.
        :type log_level: fedn.LogLevel.INFO, fedn.LogLevel.WARNING, fedn.LogLevel.ERROR
        :param type: The type of the message.
        :type type: str
        :param request: The request message.
        :type request: fedn.Request
        """
        status = fedn.Status()
        status.timestamp.GetCurrentTime()
        status.sender.name = sender_name
        status.sender.role = fedn.CLIENT
        status.log_level = log_level
        status.status = str(msg)
        status.session_id = session_id

        if type is not None:
            status.type = type

        if request is not None:
            status.data = MessageToJson(request)

        logger.info("Sending status message to combiner.")
        _ = self.connectorStub.SendStatus(status, metadata=self.metadata)

    @grpc_retry(max_retries=5, retry_interval=5)
    def send_model_metric(self, metric: fedn.ModelMetric) -> bool:
        """Send a model metric to the combiner."""
        logger.info("Sending model metric to combiner.")
        _ = self.combinerStub.SendModelMetric(metric, metadata=self.metadata)
        return True

    @grpc_retry(max_retries=5, retry_interval=5)
    def send_attributes(self, attribute: fedn.AttributeMessage) -> bool:
        """Send a attribute message to the combiner."""
        logger.debug("Sending attributes to combiner.")
        _ = self.combinerStub.SendAttributeMessage(attribute, metadata=self.metadata)
        return True

    @grpc_retry(max_retries=5, retry_interval=5)
    def send_telemetry(self, telemetry: fedn.TelemetryMessage) -> bool:
        """Send a telemetry message to the combiner."""
        logger.debug("Sending telemetry to combiner.")
        _ = self.combinerStub.SendTelemetryMessage(telemetry, metadata=self.metadata)
        return True

    @grpc_retry(max_retries=-1, retry_interval=5)
    def get_model_from_combiner(self, id: str, client_id: str, timeout: int = 20) -> Optional[BytesIO]:
        """Fetch a model from the assigned combiner.

        Downloads the model update object via a gRPC streaming channel.

        :param id: The id of the model update object.
        :type id: str
        :param client_id: The id of the client.
        :type client_id: str
        :param timeout: The timeout for the request.
        :type timeout: int
        :return: The model update object.
        :rtype: Optional[BytesIO]
        """
        data = BytesIO()
        time_start = time.time()
        request = fedn.ModelRequest(id=id)
        request.sender.client_id = client_id
        request.sender.role = fedn.CLIENT

        logger.info("Downloading model from combiner.")
        for part in self.modelStub.Download(request, metadata=self.metadata):
            if part.status == fedn.ModelStatus.IN_PROGRESS:
                data.write(part.data)

            if part.status == fedn.ModelStatus.OK:
                return data

            if part.status == fedn.ModelStatus.FAILED:
                return None

            if part.status == fedn.ModelStatus.UNKNOWN:
                if time.time() - time_start >= timeout:
                    return None
                continue
        return data

    @grpc_retry(max_retries=-1, retry_interval=5)
    def send_model_to_combiner(self, model: BytesIO, id: str) -> Optional[BytesIO]:
        """Send a model update to the assigned combiner.

        Uploads the model updated object via a gRPC streaming channel, Upload.

        :param model: The model update object.
        :type model: BytesIO
        :param id: The id of the model update object.
        :type id: str
        :return: The model update object.
        :rtype: Optional[BytesIO]
        """
        if not isinstance(model, BytesIO):
            bt = BytesIO()

            for d in model.stream(32 * 1024):
                bt.write(d)
        else:
            bt = model

        bt.seek(0, 0)

        logger.info("Uploading model to combiner.")
        result = self.modelStub.Upload(upload_request_generator(bt, id), metadata=self.metadata)
        return result

    def create_update_message(
        self,
        sender_name: str,
        model_id: str,
        model_update_id: str,
        receiver_name: str,
        receiver_role: fedn.Role,
        meta: dict,
    ) -> fedn.ModelUpdate:
        """Create an update message."""
        update = fedn.ModelUpdate()
        update.sender.name = sender_name
        update.sender.role = fedn.CLIENT
        update.sender.client_id = self.metadata[0][1]
        update.receiver.name = receiver_name
        update.receiver.role = receiver_role
        update.model_id = model_id
        update.model_update_id = model_update_id
        update.timestamp = str(datetime.now(timezone.utc))
        update.meta = json.dumps(meta)

        return update

    def create_validation_message(
        self,
        sender_name: str,
        sender_client_id: str,
        receiver_name: str,
        receiver_role: fedn.Role,
        model_id: str,
        metrics: str,
        correlation_id: str,
        session_id: str,
    ) -> fedn.ModelValidation:
        """Create a validation message."""
        validation = fedn.ModelValidation()
        validation.sender.name = sender_name
        validation.sender.client_id = sender_client_id
        validation.sender.role = fedn.CLIENT
        validation.receiver.name = receiver_name
        validation.receiver.role = receiver_role
        validation.model_id = model_id
        validation.data = metrics
        validation.timestamp.GetCurrentTime()
        validation.correlation_id = correlation_id
        validation.session_id = session_id

        return validation

    def create_prediction_message(
        self,
        sender_name: str,
        receiver_name: str,
        receiver_role: fedn.Role,
        model_id: str,
        prediction_output: str,
        correlation_id: str,
        session_id: str,
    ) -> fedn.ModelPrediction:
        """Create a prediction message."""
        prediction = fedn.ModelPrediction()
        prediction.sender.name = sender_name
        prediction.sender.role = fedn.CLIENT
        prediction.receiver.name = receiver_name
        prediction.receiver.role = receiver_role
        prediction.model_id = model_id
        prediction.data = prediction_output
        prediction.timestamp.GetCurrentTime()
        prediction.correlation_id = correlation_id
        prediction.prediction_id = session_id

        return prediction

    def create_backward_completion_message(
        self,
        sender_name: str,
        receiver_name: str,
        receiver_role: fedn.Role,
        gradient_id: str,
        session_id: str,
        meta: dict,
    ):
        completion = fedn.BackwardCompletion()
        completion.sender.name = sender_name
        completion.sender.role = fedn.CLIENT
        completion.sender.client_id = self.metadata[0][1]
        completion.receiver.name = receiver_name
        completion.receiver.role = receiver_role
        completion.gradient_id = gradient_id
        completion.timestamp.GetCurrentTime()
        completion.meta = json.dumps(meta)
        completion.session_id = session_id
        return completion

    def send_backward_completion(self, update: fedn.BackwardCompletion):
        """Send a backward completion message to the combiner."""
        try:
            logger.info("Sending backward completion to combiner.")
            _ = self.combinerStub.SendBackwardCompletion(update, metadata=self.metadata)
        except grpc.RpcError as e:
            return self._handle_grpc_error(e, "SendBackwardCompletion", lambda: self.send_backward_completion(update))
        except Exception as e:
            logger.error(f"GRPC (SendBackwardCompletion): An error occurred: {e}")
            self._handle_unknown_error(e, "SendBackwardCompletion", lambda: self.send_backward_completion(update))
        return True

    def create_metric_message(
        self, sender_name: str, sender_client_id: str, metrics: dict, step: int, model_id: str, session_id: str, round_id: str
    ) -> fedn.ModelMetric:
        """Create a metric message."""
        metric = fedn.ModelMetric()
        metric.sender.name = sender_name
        metric.sender.client_id = sender_client_id
        metric.sender.role = fedn.CLIENT
        metric.model_id = model_id
        if step is not None:
            metric.step.value = step
        metric.session_id = session_id
        metric.round_id = round_id
        metric.timestamp.GetCurrentTime()
        for key, value in metrics.items():
            metric.metrics.add(key=key, value=value)
        return metric

    @grpc_retry(max_retries=-1, retry_interval=5)
    def send_model_update(self, update: fedn.ModelUpdate) -> bool:
        """Send a model update to the combiner."""
        logger.info("Sending model update to combiner.")
        _ = self.combinerStub.SendModelUpdate(update, metadata=self.metadata)
        return True

    @grpc_retry(max_retries=-1, retry_interval=5)
    def send_model_validation(self, validation: fedn.ModelValidation) -> bool:
        """Send a model validation to the combiner."""
        logger.info("Sending model validation to combiner.")
        _ = self.combinerStub.SendModelValidation(validation, metadata=self.metadata)
        return True

    @grpc_retry(max_retries=-1, retry_interval=5)
    def send_model_prediction(self, prediction: fedn.ModelPrediction) -> bool:
        """Send a model prediction to the combiner."""
        logger.info("Sending model prediction to combiner.")
        _ = self.combinerStub.SendModelPrediction(prediction, metadata=self.metadata)
        return True

    def _disconnect(self) -> None:
        """Disconnect from the combiner."""
        self.channel.close()
        logger.info("GRPC channel closed.")

    def _reconnect(self) -> None:
        """Reconnect to the combiner."""
        self._disconnect()
        self._init_channel(self.host, self.port, self.token)
        self._init_stubs()
        logger.debug("GRPC channel reconnected.")
