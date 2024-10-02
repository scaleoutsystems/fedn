import json
import sys
import time
from datetime import datetime
from io import BytesIO
from typing import Any, Callable

import grpc
from google.protobuf.json_format import MessageToJson

import fedn.network.grpc.fedn_pb2 as fedn
import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.common.log_config import logger
from fedn.network.combiner.modelservice import upload_request_generator


class GrpcHandler:
    def __init__(self, host: str, port: int, name: str, token: str, combiner_name: str):
        self.metadata = [("client", name), ("grpc-server", combiner_name), ("authorization", token)]

        def metadata_callback(context, callback):
            callback(self.metadata, None)

        call_credentials = grpc.metadata_call_credentials(metadata_callback)
        channel_credentials = (
            grpc.composite_channel_credentials(grpc.ssl_channel_credentials(), call_credentials)
            if port == 443
            else grpc.composite_channel_credentials(grpc.local_channel_credentials(), call_credentials)
        )

        url = f"{host}:{port}"

        logger.info(f"Connecting (GRPC) to {url}")

        channel = grpc.secure_channel(url, channel_credentials) if port == 443 else grpc.insecure_channel(url)

        self.connectorStub = rpc.ConnectorStub(channel)
        self.combinerStub = rpc.CombinerStub(channel)
        self.modelStub = rpc.ModelServiceStub(channel)

    def send_heartbeats(self, client_name: str, client_id: str, update_frequency: float = 2.0):
        heartbeat = fedn.Heartbeat(sender=fedn.Client(name=client_name, role=fedn.WORKER, client_id=client_id))

        try:
            logger.info("Sending heartbeat to combiner")
            self.connectorStub.SendHeartbeat(heartbeat)
        except grpc.RpcError as e:
            status_code = e.code()

            if status_code == grpc.StatusCode.UNAVAILABLE:
                logger.error("GRPC hearbeat: combiner unavailable")

            elif status_code == grpc.StatusCode.UNAUTHENTICATED:
                details = e.details()
                if details == "Token expired":
                    logger.error("GRPC hearbeat: Token expired. Disconnecting.")
                    sys.exit("Unauthorized. Token expired. Please obtain a new token.")

        time.sleep(update_frequency)
        self.send_heartbeats(client_name=client_name, client_id=client_id, update_frequency=update_frequency)

    def listen_to_task_stream(self, client_name: str, client_id: str, callback: Callable[[Any], None]):
        """Subscribe to the model update request stream.

        :return: None
        :rtype: None
        """
        r = fedn.ClientAvailableMessage()
        r.sender.name = client_name
        r.sender.role = fedn.WORKER
        r.sender.client_id = client_id

        status_code = None

        try:
            for request in self.combinerStub.TaskStream(r, metadata=self.metadata):
                if request.sender.role == fedn.COMBINER:
                    self.send_status(
                        "Received model update request.",
                        log_level=fedn.Status.AUDIT,
                        type=fedn.StatusType.MODEL_UPDATE_REQUEST,
                        request=request,
                        sesssion_id=request.session_id,
                    )

                    logger.info(f"Received task request of type {request.type} for model_id {request.model_id}")

                    callback(request)

                    # if request.type == fedn.StatusType.MODEL_UPDATE and self.config["trainer"]:
                    #     self.inbox.put(("train", request))
                    # elif request.type == fedn.StatusType.MODEL_VALIDATION and self.config["validator"]:
                    #     self.inbox.put(("validate", request))
                    # elif request.type == fedn.StatusType.INFERENCE and self.config["validator"]:
                    #     logger.info("Received inference request for model_id {}".format(request.model_id))
                    #     presigned_url = json.loads(request.data)
                    #     presigned_url = presigned_url["presigned_url"]
                    #     logger.info("Inference presigned URL: {}".format(presigned_url))
                    #     self.inbox.put(("infer", request))
                    # else:
                    #     logger.error("Unknown request type: {}".format(request.type))

        except grpc.RpcError as e:
            status_code = e.code()
            if status_code == grpc.StatusCode.UNAVAILABLE:
                logger.warning("GRPC TaskStream: server unavailable during model update request stream. Retrying.")
                # Retry after a delay
                time.sleep(5)
                self.listen_to_task_stream(client_name=client_name, client_id=client_id, callback=callback)
            if status_code == grpc.StatusCode.UNAUTHENTICATED:
                details = e.details()
                if details == "Token expired":
                    logger.warning("GRPC TaskStream: Token expired. Reconnecting.")
                    # TODO: Disconnect?
                    # self.disconnect()

            if status_code == grpc.StatusCode.CANCELLED:
                # Expected if the client is disconnected
                logger.critical("GRPC TaskStream: Client disconnected from combiner. Trying to reconnect.")

            else:
                # Log the error and continue
                logger.error(f"GRPC TaskStream: An error occurred during model update request stream: {e}")

    def send_status(self, msg: str, log_level=fedn.Status.INFO, type=None, request=None, sesssion_id: str = None):
        """Send status message.

        :param msg: The message to send.
        :type msg: str
        :param log_level: The log level of the message.
        :type log_level: fedn.Status.INFO, fedn.Status.WARNING, fedn.Status.ERROR
        :param type: The type of the message.
        :type type: str
        :param request: The request message.
        :type request: fedn.Request
        """
        status = fedn.Status()
        status.timestamp.GetCurrentTime()
        # TODO: name...
        status.sender.name = "self.name"
        status.sender.role = fedn.WORKER
        status.log_level = log_level
        status.status = str(msg)
        status.session_id = sesssion_id

        if type is not None:
            status.type = type

        if request is not None:
            status.data = MessageToJson(request)

        try:
            _ = self.connectorStub.SendStatus(status, metadata=self.metadata)
        except grpc.RpcError as e:
            status_code = e.code()
            if status_code == grpc.StatusCode.UNAVAILABLE:
                logger.warning("GRPC SendStatus: server unavailable during send status.")
            if status_code == grpc.StatusCode.UNAUTHENTICATED:
                details = e.details()
                if details == "Token expired":
                    logger.warning("GRPC SendStatus: Token expired.")

    def get_model_from_combiner(self, id: str, client_name: str, timeout: int = 20) -> BytesIO:
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
        request.sender.name = client_name
        request.sender.role = fedn.WORKER

        try:
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
            logger.critical(f"GRPC: An error occurred during model download: {e}")

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
            result = self.modelStub.Upload(upload_request_generator(bt, id), metadata=self.metadata)
        except grpc.RpcError as e:
            logger.critical(f"GRPC: An error occurred during model upload: {e}")

        return result

    def send_model_update(self,
        sender_name: str,
        sender_role: fedn.Role,
        model_id: str,
        model_update_id: str,
        receiver_name: str,
        receiver_role: fedn.Role,
        meta: dict
    ):
        update = fedn.ModelUpdate()
        update.sender.name = sender_name
        update.sender.role = sender_role
        update.receiver.name = receiver_name
        update.receiver.role = receiver_role
        update.model_id = model_id
        update.model_update_id = model_update_id
        update.timestamp = str(datetime.now())
        update.meta = json.dumps(meta)

        try:
            _ = self.combinerStub.SendModelUpdate(update, metadata=self.metadata)
        except grpc.RpcError as e:
            status_code = e.code()
            if status_code == grpc.StatusCode.UNAVAILABLE:
                logger.warning("GRPC SendModelUpdate: server unavailable during send model update.")
            if status_code == grpc.StatusCode.UNAUTHENTICATED:
                details = e.details()
                if details == "Token expired":
                    logger.warning("GRPC SendModelUpdate: Token expired.")
            return False
        except Exception as e:
            logger.error(f"GRPC SendModelUpdate: An error occurred: {e}")
            return False

        return True

    def send_model_validation(self,
        sender_name: str,
        receiver_name: str,
        receiver_role: fedn.Role,
        model_id: str,
        metrics: str,
        correlation_id: str,
        session_id: str
    ) -> bool:
        validation = fedn.ModelValidation()
        validation.sender.name = sender_name
        validation.sender.role = fedn.WORKER
        validation.receiver.name = receiver_name
        validation.receiver.role = receiver_role
        validation.model_id = model_id
        validation.data = metrics
        validation.timestamp.GetCurrentTime()
        validation.correlation_id = correlation_id
        validation.session_id = session_id


        try:
            _ = self.combinerStub.SendModelValidation(validation, metadata=self.metadata)
        except grpc.RpcError as e:
            status_code = e.code()
            if status_code == grpc.StatusCode.UNAVAILABLE:
                logger.warning("GRPC SendModelValidation: server unavailable during send model validation.")
            if status_code == grpc.StatusCode.UNAUTHENTICATED:
                details = e.details()
                if details == "Token expired":
                    logger.warning("GRPC SendModelValidation: Token expired.")
            return False
        except Exception as e:
            logger.error(f"GRPC SendModelValidation: An error occurred: {e}")
            return False

        return True
