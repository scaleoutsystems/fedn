import json
import os
import socket
import time
from datetime import datetime
from io import BytesIO
from typing import Any, Callable

import grpc
from cryptography.hazmat.primitives.serialization import Encoding
from google.protobuf.json_format import MessageToJson
from OpenSSL import SSL

import fedn.network.grpc.fedn_pb2 as fedn
import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.common.config import FEDN_AUTH_SCHEME
from fedn.common.log_config import logger
from fedn.network.combiner.modelservice import upload_request_generator

# Keepalive settings: these help keep the connection open for long-lived clients
KEEPALIVE_TIME_MS = 1 * 1000  # send keepalive ping every 60 seconds
KEEPALIVE_TIMEOUT_MS = 30 * 1000  # wait 20 seconds for keepalive ping ack before considering connection dead
KEEPALIVE_PERMIT_WITHOUT_CALLS = True  # allow keepalive pings even when there are no RPCs
MAX_CONNECTION_IDLE_MS = 30000
MAX_CONNECTION_AGE_GRACE_MS = "INT_MAX"  # keep connection open indefinitely
CLIENT_IDLE_TIMEOUT_MS = 30000

GRPC_OPTIONS = [
    ("grpc.keepalive_time_ms", KEEPALIVE_TIME_MS),
    ("grpc.keepalive_timeout_ms", KEEPALIVE_TIMEOUT_MS),
    ("grpc.keepalive_permit_without_calls", KEEPALIVE_PERMIT_WITHOUT_CALLS),
    ("grpc.http2.max_pings_without_data", 0),  # unlimited pings without data
    ("grpc.max_connection_idle_ms", MAX_CONNECTION_IDLE_MS),
    ("grpc.max_connection_age_grace_ms", MAX_CONNECTION_AGE_GRACE_MS),
    ("grpc.client_idle_timeout_ms", CLIENT_IDLE_TIMEOUT_MS),
]


class GrpcAuth(grpc.AuthMetadataPlugin):
    def __init__(self, key):
        self._key = key

    def __call__(self, context, callback):
        callback((("authorization", f"{FEDN_AUTH_SCHEME} {self._key}"),), None)


def _get_ssl_certificate(domain, port=443):
    context = SSL.Context(SSL.TLSv1_2_METHOD)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((domain, port))
    ssl_sock = SSL.Connection(context, sock)
    ssl_sock.set_tlsext_host_name(domain.encode())
    ssl_sock.set_connect_state()
    ssl_sock.do_handshake()
    cert = ssl_sock.get_peer_certificate()
    ssl_sock.close()
    sock.close()
    cert = cert.to_cryptography().public_bytes(Encoding.PEM).decode()
    return cert


class GrpcHandler:
    def __init__(self, host: str, port: int, name: str, token: str, combiner_name: str):
        self.metadata = [
            ("client", name),
            ("grpc-server", combiner_name),
        ]
        self.host = host
        self.port = port
        self.token = token

        self._init_channel(host, port, token)

        self._init_stubs()

    def _init_stubs(self):
        self.connectorStub = rpc.ConnectorStub(self.channel)
        self.combinerStub = rpc.CombinerStub(self.channel)
        self.modelStub = rpc.ModelServiceStub(self.channel)

    def _init_channel(self, host: str, port: int, token: str):
        if port == 443:
            self._init_secure_channel(host, port, token)
        else:
            self._init_insecure_channel(host, port)

    def _init_secure_channel(self, host: str, port: int, token: str):
        url = f"{host}:{port}"
        logger.info(f"Connecting (GRPC) to {url}")

        if os.getenv("FEDN_GRPC_ROOT_CERT_PATH"):
            logger.info("Using root certificate from environment variable for GRPC channel.")
            with open(os.environ["FEDN_GRPC_ROOT_CERT_PATH"], "rb") as f:
                credentials = grpc.ssl_channel_credentials(f.read())
            self.channel = grpc.secure_channel("{}:{}".format(host, str(port)), credentials)
            return

        logger.info(f"Fetching SSL certificate for {host}")
        cert = _get_ssl_certificate(host, port)
        credentials = grpc.ssl_channel_credentials(cert.encode("utf-8"))
        auth_creds = grpc.metadata_call_credentials(GrpcAuth(token))
        self.channel = grpc.secure_channel(
            "{}:{}".format(host, str(port)),
            grpc.composite_channel_credentials(credentials, auth_creds),
            options=GRPC_OPTIONS,
        )

    def _init_insecure_channel(self, host: str, port: int):
        url = f"{host}:{port}"
        logger.info(f"Connecting (GRPC) to {url}")
        self.channel = grpc.insecure_channel(
            url,
            options=GRPC_OPTIONS,
        )

    def heartbeat(self, client_name: str, client_id: str):
        """Send a heartbeat to the combiner.

        :return: Response from the combiner.
        :rtype: fedn.Response
        """
        heartbeat = fedn.Heartbeat(sender=fedn.Client(name=client_name, role=fedn.CLIENT, client_id=client_id))

        try:
            logger.info("Sending heartbeat to combiner")
            response = self.connectorStub.SendHeartbeat(heartbeat, metadata=self.metadata)
        except grpc.RpcError as e:
            logger.error(f"GRPC (SendHeartbeat): An error occurred: {e}")
            raise e
        except Exception as e:
            logger.error(f"GRPC (SendHeartbeat): An error occurred: {e}")
            raise e
        return response

    def send_heartbeats(self, client_name: str, client_id: str, update_frequency: float = 2.0):
        send_hearbeat = True
        while send_hearbeat:
            try:
                response = self.heartbeat(client_name, client_id)
            except grpc.RpcError as e:
                return self._handle_grpc_error(e, "SendHeartbeat", lambda: self.send_heartbeats(client_name, client_id, update_frequency))
            except Exception as e:
                return self._handle_unknown_error(e, "SendHeartbeat", lambda: self.send_heartbeats(client_name, client_id, update_frequency))
            if isinstance(response, fedn.Response):
                logger.info("Heartbeat successful.")
            else:
                logger.error("Heartbeat failed.")
                send_hearbeat = False
            time.sleep(update_frequency)

    def listen_to_task_stream(self, client_name: str, client_id: str, callback: Callable[[Any], None]):
        """Subscribe to the model update request stream.

        :return: None
        :rtype: None
        """
        r = fedn.ClientAvailableMessage()
        r.sender.name = client_name
        r.sender.role = fedn.CLIENT
        r.sender.client_id = client_id

        try:
            logger.info("Listening to task stream.")
            for request in self.combinerStub.TaskStream(r, metadata=self.metadata):
                if request.sender.role == fedn.COMBINER:
                    self.send_status(
                        "Received request from combiner.",
                        log_level=fedn.LogLevel.AUDIT,
                        type=request.type,
                        request=request,
                        sesssion_id=request.session_id,
                        sender_name=client_name,
                    )

                    logger.info(f"Received task request of type {request.type} for model_id {request.model_id}")

                    callback(request)

        except grpc.RpcError as e:
            logger.error(f"GRPC (TaskStream): An error occurred: {e}")
            return self._handle_grpc_error(e, "TaskStream", lambda: self.listen_to_task_stream(client_name, client_id, callback))
        except Exception as e:
            logger.error(f"GRPC (TaskStream): An error occurred: {e}")
            self._handle_unknown_error(e, "TaskStream", lambda: self.listen_to_task_stream(client_name, client_id, callback))

    def send_status(self, msg: str, log_level=fedn.LogLevel.INFO, type=None, request=None, sesssion_id: str = None, sender_name: str = None):
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
        status.session_id = sesssion_id

        if type is not None:
            status.type = type

        if request is not None:
            status.data = MessageToJson(request)

        try:
            logger.info("Sending status message to combiner.")
            _ = self.connectorStub.SendStatus(status, metadata=self.metadata)
        except grpc.RpcError as e:
            return self._handle_grpc_error(e, "SendStatus", lambda: self.send_status(msg, log_level, type, request, sesssion_id, sender_name))
        except Exception as e:
            logger.error(f"GRPC (SendStatus): An error occurred: {e}")
            self._handle_unknown_error(e, "SendStatus", lambda: self.send_status(msg, log_level, type, request, sesssion_id, sender_name))

    def get_model_from_combiner(self, id: str, client_id: str, timeout: int = 20) -> BytesIO:
        """Fetch a model from the assigned combiner.
        Downloads the model update object via a gRPC streaming channel.

        :param id: The id of the model update object.
        :type id: str
        :return: The model update object.
        :rtype: BytesIO
        """
        data = BytesIO()
        time_start = time.time()
        request = fedn.ModelRequest(id=id)
        request.sender.client_id = client_id
        request.sender.role = fedn.CLIENT

        try:
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
        except grpc.RpcError as e:
            return self._handle_grpc_error(e, "Download", lambda: self.get_model_from_combiner(id, client_id, timeout))
        except Exception as e:
            logger.error(f"GRPC (Download): An error occurred: {e}")
            self._handle_unknown_error(e, "Download", lambda: self.get_model_from_combiner(id, client_id, timeout))
        return data

    def send_model_to_combiner(self, model: BytesIO, id: str):
        """Send a model update to the assigned combiner.
        Uploads the model updated object via a gRPC streaming channel, Upload.

        :param model: The model update object.
        :type model: BytesIO
        :param id: The id of the model update object.
        :type id: str
        :return: The model update object.
        :rtype: BytesIO
        """
        if not isinstance(model, BytesIO):
            bt = BytesIO()

            for d in model.stream(32 * 1024):
                bt.write(d)
        else:
            bt = model

        bt.seek(0, 0)

        try:
            logger.info("Uploading model to combiner.")
            result = self.modelStub.Upload(upload_request_generator(bt, id), metadata=self.metadata)
        except grpc.RpcError as e:
            return self._handle_grpc_error(e, "Upload", lambda: self.send_model_to_combiner(model, id))
        except Exception as e:
            logger.error(f"GRPC (Upload): An error occurred: {e}")
            self._handle_unknown_error(e, "Upload", lambda: self.send_model_to_combiner(model, id))
        return result

    def create_update_message(
        self,
        sender_name: str,
        model_id: str,
        model_update_id: str,
        receiver_name: str,
        receiver_role: fedn.Role,
        meta: dict,
    ):
        update = fedn.ModelUpdate()
        update.sender.name = sender_name
        update.sender.role = fedn.CLIENT
        update.sender.client_id = self.metadata[0][1]
        update.receiver.name = receiver_name
        update.receiver.role = receiver_role
        update.model_id = model_id
        update.model_update_id = model_update_id
        update.timestamp = str(datetime.now())
        update.meta = json.dumps(meta)

        return update

    def create_validation_message(
        self,
        sender_name: str,
        receiver_name: str,
        receiver_role: fedn.Role,
        model_id: str,
        metrics: str,
        correlation_id: str,
        session_id: str,
    ):
        validation = fedn.ModelValidation()
        validation.sender.name = sender_name
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
    ):
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

    def send_model_update(self, update: fedn.ModelUpdate):
        try:
            logger.info("Sending model update to combiner.")
            _ = self.combinerStub.SendModelUpdate(update, metadata=self.metadata)
        except grpc.RpcError as e:
            return self._handle_grpc_error(e, "SendModelUpdate", lambda: self.send_model_update(update))
        except Exception as e:
            logger.error(f"GRPC (SendModelUpdate): An error occurred: {e}")
            self._handle_unknown_error(e, "SendModelUpdate", lambda: self.send_model_update(update))
        return True

    def send_model_validation(self, validation: fedn.ModelValidation) -> bool:
        try:
            logger.info("Sending model validation to combiner.")
            _ = self.combinerStub.SendModelValidation(validation, metadata=self.metadata)
        except grpc.RpcError as e:
            return self._handle_grpc_error(
                e,
                "SendModelValidation",
                lambda: self.send_model_validation(validation),
            )
        except Exception as e:
            logger.error(f"GRPC (SendModelValidation): An error occurred: {e}")
            self._handle_unknown_error(e, "SendModelValidation", lambda: self.send_model_validation(validation))
        return True

    def send_model_prediction(self, prediction: fedn.ModelPrediction) -> bool:
        try:
            logger.info("Sending model prediction to combiner.")
            _ = self.combinerStub.SendModelPrediction(prediction, metadata=self.metadata)
        except grpc.RpcError as e:
            return self._handle_grpc_error(
                e,
                "SendModelPrediction",
                lambda: self.send_model_prediction(prediction),
            )
        except Exception as e:
            logger.error(f"GRPC (SendModelPrediction): An error occurred: {e}")
            self._handle_unknown_error(e, "SendModelPrediction", lambda: self.send_model_prediction(prediction))
        return True

    def _handle_grpc_error(self, e, method_name: str, sender_function: Callable):
        status_code = e.code()
        if status_code == grpc.StatusCode.UNAVAILABLE:
            logger.warning(f"GRPC ({method_name}): server unavailable. Retrying in 5 seconds.")
            time.sleep(5)
            return sender_function()
        elif status_code == grpc.StatusCode.CANCELLED:
            logger.warning(f"GRPC ({method_name}): connection cancelled. Retrying in 5 seconds.")
            time.sleep(5)
            return sender_function()
        elif status_code == grpc.StatusCode.UNAUTHENTICATED:
            details = e.details()
            if details == "Token expired":
                logger.warning(f"GRPC ({method_name}): Token expired.")
                raise e
        elif status_code == grpc.StatusCode.UNKNOWN:
            logger.warning(f"GRPC ({method_name}): An unknown error occurred: {e}.")
            details = e.details()
            if details == "Stream removed":
                logger.warning(f"GRPC ({method_name}): Stream removed. Reconnecting")
                self._disconnect()
                self._init_channel(self.host, self.port, self.token)
                self._init_stubs()
                return sender_function()
            raise e
        self._disconnect()
        logger.error(f"GRPC ({method_name}): An error occurred: {e}")
        raise e

    def _handle_unknown_error(self, e, method_name: str, sender_function: Callable):
        # Try to reconnect
        logger.warning(f"GRPC ({method_name}): An unknown error occurred: {e}.")
        if isinstance(e, ValueError):
            # ValueError is raised when the channel is closed
            self._disconnect()
            logger.warning(f"GRPC ({method_name}): Reconnecting to channel.")
            # recreate the channel
            self._init_channel(self.host, self.port, self.token)
            self._init_stubs()
            return sender_function()
        else:
            raise e

    def _disconnect(self):
        """Disconnect from the combiner."""
        self.channel.close()
        logger.info("GRPC channel closed.")
