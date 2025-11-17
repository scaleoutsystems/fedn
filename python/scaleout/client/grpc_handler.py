"""GrpcHandler class for handling GRPC connections and operations."""

import json
import os
import random
import time
from functools import wraps
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

from scaleoututil.grpc.clientrequest import ClientRequestType
from scaleoututil.utils.dist import get_version
import grpc

import scaleoututil.grpc.scaleout_pb2 as scaleout_msg
import scaleoututil.grpc.scaleout_pb2_grpc as rpc
from scaleoututil.config import SCALEOUT_AUTH_SCHEME
from scaleoututil.logging import FednLogger
from scaleoututil.utils.checksum import compute_checksum_from_stream

if TYPE_CHECKING:
    from scaleout.client.fedn_client import FednClient  # not-floating-import

# Keepalive settings: these help keep the connection open for long-lived clients
CHUNK_SIZE = 32 * 1024  # 32KB

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


def upload_request_generator(model_stream: BytesIO):
    """Generator function for model upload requests for the client

    :param mdl: The model update object.
    :type mdl: BytesIO
    :return: A model update request.
    :rtype: scaleout.FileChunk
    """
    while True:
        b = model_stream.read(CHUNK_SIZE)
        if b:
            yield scaleout_msg.FileChunk(data=b)
        else:
            break


class GrpcAuth(grpc.AuthMetadataPlugin):
    """GRPC authentication plugin."""

    def __init__(self, key: str) -> None:
        """Initialize GrpcAuth with a key."""
        self._key = key

    def __call__(self, context: grpc.AuthMetadataContext, callback: grpc.AuthMetadataPluginCallback) -> None:
        """Add authorization metadata to the GRPC call."""
        callback((("authorization", f"{SCALEOUT_AUTH_SCHEME} {self._key}"),), None)


class RetryException(Exception):
    pass


def grpc_retry(
    max_retries: int = 3,
    base_retry_interval: float = 1.0,
    backoff: float = 1.5,
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
            backoff_factor = 1.0 / backoff  # so that it is 1.0 on the first try
            while max_retries > retries or max_retries == -1:
                retries += 1
                backoff_factor *= backoff

                # Reset backoff factor if the last try was more than 16 times the retry interval ago
                # This is to prevent the backoff factor from growing too large
                # if the server is down for a long time and then comes back up
                this_try = time.time()
                if this_try - last_try > 16 * base_retry_interval:
                    backoff_factor = 1.0
                last_try = this_try

                retry_interval = base_retry_interval * (backoff_factor + random.uniform(-0.5, 0.5))
                try:
                    return func(self, *args, **kwargs)
                except grpc.RpcError as e:
                    status_code = e.code()
                    if status_code == grpc.StatusCode.UNAVAILABLE:
                        FednLogger().warning(f"GRPC ({func.__name__}): Server unavailable. Retrying in {retry_interval:.2f} seconds.")
                        FednLogger().debug(f"GRPC ({func.__name__}): Error details: {e.details()}")
                        self._reconnect_channel()
                        time.sleep(retry_interval)
                        continue
                    if status_code == grpc.StatusCode.CANCELLED:
                        FednLogger().warning(f"GRPC ({func.__name__}): Connection cancelled. Retrying in approx {retry_interval:.2f} seconds.")
                        FednLogger().debug(f"GRPC ({func.__name__}): Error details: {e.details()}")
                        time.sleep(retry_interval)
                        continue
                    if status_code == grpc.StatusCode.UNKNOWN:
                        details = e.details()
                        if details == "Stream removed":
                            FednLogger().warning(f"GRPC ({func.__name__}): Stream removed. Retrying in approx {retry_interval:.2f} seconds.")
                            self._reconnect_channel()
                            time.sleep(retry_interval)
                            continue
                        raise e
                    raise e
                except Exception as e:
                    FednLogger().warning(f"GRPC ({func.__name__}): An unknown error occurred: {e}.")
                    if isinstance(e, ValueError):
                        FednLogger().warning(f"GRPC ({func.__name__}): Retrying in approx {retry_interval:.2f} seconds.")
                        self._reconnect_channel()
                        time.sleep(retry_interval)
                        continue
                    raise e

            FednLogger().error(f"GRPC ({func.__name__}): Max retries exceeded.")
            raise RetryException("Max retries exceeded")

        return wrapper

    return decorator


def compute_checksum(model_stream: BytesIO) -> str:
    """Compute the checksum of a model.

    :param model: The model to compute the checksum for.
    :type model: BytesIO
    :return: The checksum of the model.
    :rtype: str
    """
    return compute_checksum_from_stream(model_stream)


class GrpcHandler:
    """Handler for GRPC connections and operations."""

    def __init__(self, client: "FednClient", host: str, port: int, token: str) -> None:
        """Initialize the GrpcHandler."""
        self.client = client

        os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"  # Actively disable fork support in GRPC

        self.host = host
        self.port = port
        self.token = token

        self._init_channel(host, port, token)
        self._init_stubs()

    @property
    def client_id(self) -> str:
        return self.client.client_id

    @property
    def metadata(self) -> list:
        val = [("client", self.client.client_id), ("client-id", self.client.client_id), ("name", self.client.name)]

        # Add authorization token if available
        if hasattr(self, "_auth_token") and self._auth_token:
            val.append(("authorization", f"{SCALEOUT_AUTH_SCHEME} {self._auth_token}"))

        return val

    def _init_stubs(self) -> None:
        """Initialize GRPC stubs."""
        self.combinerStub = rpc.CombinerClientStub(self.channel)
        self.modelStub = rpc.ModelServiceStub(self.channel)

    def _init_channel(self, host: str, port: int, token: str) -> None:
        """Initialize the GRPC channel."""
        if port == GRPC_SECURE_PORT:
            self._init_secure_channel(host, port, token)
        else:
            self._init_insecure_channel(host, port, token)

    def _init_secure_channel(self, host: str, port: int, token: str) -> None:
        """Initialize a secure GRPC channel."""
        url = f"{host}:{port}"
        FednLogger().info(f"Connecting (GRPC) to {url}")

        if os.getenv("SCALEOUT_GRPC_ROOT_CERT_PATH"):
            FednLogger().info("Using root certificate from environment variable for GRPC channel.")
            with open(os.environ["SCALEOUT_GRPC_ROOT_CERT_PATH"], "rb") as f:
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

    def _init_insecure_channel(self, host: str, port: int, token: Optional[str] = None) -> None:
        """Initialize an insecure GRPC channel."""
        url = f"{host}:{port}"
        FednLogger().info(f"Connecting (GRPC) to {url}")
        self.channel = grpc.insecure_channel(
            url,
            options=GRPC_OPTIONS,
        )
        # Store token to add to metadata in each call
        self._auth_token = token if token else None
        if self._auth_token:
            FednLogger().info("Token will be added to metadata for authentication.")

    def heartbeat(self) -> scaleout_msg.Response:
        """Send a heartbeat to the combiner.

        :return: Response from the combiner.
        :rtype: scaleout.Response
        """
        heartbeat = scaleout_msg.Heartbeat(client_id=self.client_id)

        response = self.combinerStub.SendHeartbeat(heartbeat, metadata=self.metadata)

        return response

    @grpc_retry(max_retries=-1)
    def send_heartbeats(self, client_name: str, client_id: str, update_frequency: float = 2.0) -> None:
        """Send heartbeats to the combiner at regular intervals."""
        send_heartbeat = True
        while send_heartbeat:
            response = self.heartbeat()
            time.sleep(update_frequency)
            if isinstance(response, scaleout_msg.Response):
                pass
            else:
                FednLogger().error("Heartbeat failed.")
                send_heartbeat = False

    @grpc_retry(max_retries=-1)
    def listen_to_task_stream(self, client_id: str, callback: Callable[[Any], None]) -> None:
        """Subscribe to the model update request stream."""
        r = scaleout_msg.ClientAvailableMessage()
        r.client_id = client_id

        FednLogger().info("Listening to task stream.")
        request: scaleout_msg.TaskRequest
        for request in self.combinerStub.TaskStream(r, metadata=self.metadata):
            FednLogger().info(f"Received task request of type {request.type} for model_id {request.model_id}")
            callback(request)

    @grpc_retry(max_retries=-1)
    def PollAndReport(self, report: scaleout_msg.ActivityReport) -> scaleout_msg.TaskRequest:
        return self.combinerStub.PollAndReport(report, metadata=self.metadata)

    @grpc_retry(max_retries=5)
    def send_status(
        self,
        msg: str,
        log_level: scaleout_msg.LogLevel = scaleout_msg.LogLevel.INFO,
        type: Optional[str] = None,
    ) -> None:
        """Send status message.

        :param msg: The message to send.
        :type msg: str
        :param log_level: The log level of the message.
        :type log_level: scaleout.LogLevel.INFO, scaleout.LogLevel.WARNING, scaleout.LogLevel.ERROR
        :param type: The type of the message.
        :type type: str
        :param request: The request message.
        :type request: scaleout.Request
        """
        status = scaleout_msg.Status()
        status.timestamp.GetCurrentTime()
        status.client_id = self.client_id
        status.log_level = log_level
        status.status = str(msg)

        if type is not None:
            status.type = type

        FednLogger().info("Sending status message to combiner.")
        _ = self.combinerStub.SendStatus(status, metadata=self.metadata)

    @grpc_retry(max_retries=5)
    def send_model_metric(self, metric: scaleout_msg.ModelMetric) -> bool:
        """Send a model metric to the combiner."""
        FednLogger().info("Sending model metric to combiner.")
        _ = self.combinerStub.SendModelMetric(metric, metadata=self.metadata)
        return True

    @grpc_retry(max_retries=5)
    def send_attributes(self, attribute: scaleout_msg.AttributeMessage) -> bool:
        """Send a attribute message to the combiner."""
        FednLogger().debug("Sending attributes to combiner.")
        _ = self.combinerStub.SendAttributeMessage(attribute, metadata=self.metadata)
        return True

    @grpc_retry(max_retries=5)
    def send_telemetry(self, telemetry: scaleout_msg.TelemetryMessage) -> bool:
        """Send a telemetry message to the combiner."""
        FednLogger().debug("Sending telemetry to combiner.")
        _ = self.combinerStub.SendTelemetryMessage(telemetry, metadata=self.metadata)
        return True

    @grpc_retry(max_retries=-1)
    def connect(self) -> scaleout_msg.Response:
        """Connect the client to the combiner."""
        request = scaleout_msg.ClientAnnounceRequest()
        request.client_id = self.client_id
        request.type = ClientRequestType.Connect.value
        FednLogger().info("Connecting client to combiner.")
        response = self.combinerStub.Announce(request, metadata=self.metadata)
        return response

    def disconnect(self) -> scaleout_msg.Response:
        """Disconnect the client from the combiner."""
        request = scaleout_msg.ClientAnnounceRequest()
        request.client_id = self.client_id
        request.type = ClientRequestType.Disconnect.value
        FednLogger().info("Disconnecting client from combiner.")
        response = self.combinerStub.Announce(request, metadata=self.metadata)
        return response

    def check_version_compatibility(self) -> Tuple[bool, str, str]:
        """Check version compatibility with the combiner."""
        request = scaleout_msg.ClientAnnounceRequest()
        request.client_id = self.client_id
        request.type = ClientRequestType.VersionCheck.value
        request.parameters = json.dumps({"client-version": get_version()})
        response: scaleout_msg.Response = self.combinerStub.Announce(request, metadata=self.metadata)
        params = json.loads(response.parameters) if response.parameters else {}
        success = params.get("success", False)
        server_version = params.get("version", "")
        msg = response.response
        return success, server_version, msg

    @grpc_retry(max_retries=-1)
    def get_model_from_combiner(self, model_id: str) -> Optional[BytesIO]:
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
        request = scaleout_msg.ModelRequest(model_id=model_id)

        FednLogger().info("Downloading model from combiner.")
        part_iterator = self.modelStub.Download(request, metadata=self.metadata)
        for part in part_iterator:
            data.write(part.data)
        metadata = dict(part_iterator.trailing_metadata())
        server_checksum = metadata.get("checksum")
        if server_checksum:
            data.seek(0, 0)
            file_checksum = compute_checksum(data)
            if file_checksum != server_checksum:
                FednLogger().error("Checksum mismatch! File is corrupted!")
                # Uncomment the following lines if you want to compute the checksum
                # and compare it with the metadata checksum
                raise ValueError("Checksum mismatch! File is corrupted!")
            else:
                FednLogger().info("Checksum match! File is valid!")

        data.seek(0, 0)
        return data

    @grpc_retry(max_retries=-1)
    def send_model_to_combiner(self, model: BytesIO, model_id: str) -> Optional[BytesIO]:
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
            byte_stream = BytesIO()

            for d in model.stream(32 * 1024):
                byte_stream.write(d)
        else:
            byte_stream = model

        byte_stream.seek(0, 0)
        file_checksum = compute_checksum(byte_stream)
        byte_stream.seek(0, 0)

        FednLogger().info("Uploading model to combiner.")
        metadata = [*self.metadata, ("model-id", model_id), ("checksum", file_checksum)]
        result = self.modelStub.Upload(upload_request_generator(byte_stream), metadata=metadata)

        return result

    def create_update_message(
        self,
        model_id: str,
        model_update_id: str,
        correlation_id: str,
        round_id: str,
        session_id: str,
        meta: dict,
    ) -> scaleout_msg.ModelUpdate:
        """Create an update message."""
        update = scaleout_msg.ModelUpdate()
        update.client_id = self.client_id
        update.model_id = model_id
        update.model_update_id = model_update_id
        update.correlation_id = correlation_id
        update.round_id = round_id
        update.session_id = session_id
        update.timestamp.GetCurrentTime()
        update.meta = json.dumps(meta)

        return update

    def create_validation_message(
        self,
        model_id: str,
        metrics: str,
        correlation_id: str,
        session_id: str,
    ) -> scaleout_msg.ModelValidation:
        """Create a validation message."""
        validation = scaleout_msg.ModelValidation()
        validation.client_id = self.client_id
        validation.model_id = model_id
        validation.data = metrics
        validation.timestamp.GetCurrentTime()
        validation.correlation_id = correlation_id
        validation.session_id = session_id

        return validation

    def create_prediction_message(
        self,
        model_id: str,
        prediction_output: str,
        correlation_id: str,
        session_id: str,
    ) -> scaleout_msg.ModelPrediction:
        """Create a prediction message."""
        prediction = scaleout_msg.ModelPrediction()
        prediction.client_id = self.client_id
        prediction.model_id = model_id
        prediction.data = prediction_output
        prediction.timestamp.GetCurrentTime()
        prediction.correlation_id = correlation_id
        prediction.session_id = session_id

        return prediction

    def create_backward_completion_message(
        self,
        gradient_id: str,
        session_id: str,
    ):
        completion = scaleout_msg.BackwardCompletion()
        completion.client_id = self.client_id
        completion.gradient_id = gradient_id
        completion.timestamp.GetCurrentTime()
        completion.session_id = session_id
        return completion

    def send_backward_completion(self, update: scaleout_msg.BackwardCompletion):
        """Send a backward completion message to the combiner."""
        try:
            FednLogger().info("Sending backward completion to combiner.")
            _ = self.combinerStub.SendBackwardCompletion(update, metadata=self.metadata)
        except grpc.RpcError as e:
            return self._handle_grpc_error(e, "SendBackwardCompletion", lambda: self.send_backward_completion(update))
        except Exception as e:
            FednLogger().error(f"GRPC (SendBackwardCompletion): An error occurred: {e}")
            self._handle_unknown_error(e, "SendBackwardCompletion", lambda: self.send_backward_completion(update))
        return True

    def create_metric_message(self, metrics: dict, step: int, model_id: str, session_id: str, round_id: str) -> scaleout_msg.ModelMetric:
        """Create a metric message."""
        metric = scaleout_msg.ModelMetric()
        metric.client_id = self.client_id
        metric.model_id = model_id
        if step is not None:
            metric.step.value = step
        metric.session_id = session_id
        metric.round_id = round_id
        metric.timestamp.GetCurrentTime()
        for key, value in metrics.items():
            metric.metrics.add(key=key, value=value)
        return metric

    @grpc_retry(max_retries=-1)
    def send_model_update(self, update: scaleout_msg.ModelUpdate) -> bool:
        """Send a model update to the combiner."""
        FednLogger().info("Sending model update to combiner.")
        _ = self.combinerStub.SendModelUpdate(update, metadata=self.metadata)
        return True

    @grpc_retry(max_retries=-1)
    def send_model_validation(self, validation: scaleout_msg.ModelValidation) -> bool:
        """Send a model validation to the combiner."""
        FednLogger().info("Sending model validation to combiner.")
        _ = self.combinerStub.SendModelValidation(validation, metadata=self.metadata)
        return True

    @grpc_retry(max_retries=-1)
    def send_model_prediction(self, prediction: scaleout_msg.ModelPrediction) -> bool:
        """Send a model prediction to the combiner."""
        FednLogger().info("Sending model prediction to combiner.")
        _ = self.combinerStub.SendModelPrediction(prediction, metadata=self.metadata)
        return True

    def _disconnect_channel(self) -> None:
        """Disconnect from the combiner."""
        self.channel.close()
        FednLogger().info("GRPC channel closed.")

    def _reconnect_channel(self) -> None:
        """Reconnect to the combiner."""
        self._disconnect_channel()
        self._init_channel(self.host, self.port, self.token)
        self._init_stubs()
        FednLogger().debug("GRPC channel reconnected.")


class GrpcConnectionOptions:
    """Options for configuring the GRPC connection."""

    def __init__(self, host: str, port: int, status: str = "", fqdn: str = "", package: str = "", ip: str = "", helper_type: str = "") -> None:
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
