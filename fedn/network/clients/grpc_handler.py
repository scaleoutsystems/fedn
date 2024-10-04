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

        self.channel = grpc.secure_channel(url, channel_credentials) if port == 443 else grpc.insecure_channel(url)

        self.connectorStub = rpc.ConnectorStub(self.channel)
        self.combinerStub = rpc.CombinerStub(self.channel)
        self.modelStub = rpc.ModelServiceStub(self.channel)

    def send_heartbeats(self, client_name: str, client_id: str, update_frequency: float = 2.0):
        heartbeat = fedn.Heartbeat(sender=fedn.Client(name=client_name, role=fedn.WORKER, client_id=client_id))

        send_hearbeat = True
        while send_hearbeat:
            try:
                logger.info("Sending heartbeat to combiner")
                self.connectorStub.SendHeartbeat(heartbeat)
            except grpc.RpcError as e:
                return self._handle_grpc_error(e, "SendHeartbeat", lambda: self.send_heartbeats(client_name, client_id, update_frequency))
            except Exception as e:
                logger.error(f"GRPC (SendHeartbeat): An error occurred: {e}")
                self._disconnect()

            time.sleep(update_frequency)

    def listen_to_task_stream(self, client_name: str, client_id: str, callback: Callable[[Any], None]):
        """Subscribe to the model update request stream.

        :return: None
        :rtype: None
        """
        r = fedn.ClientAvailableMessage()
        r.sender.name = client_name
        r.sender.role = fedn.WORKER
        r.sender.client_id = client_id

        try:
            logger.info("Listening to task stream.")
            for request in self.combinerStub.TaskStream(r, metadata=self.metadata):
                if request.sender.role == fedn.COMBINER:
                    self.send_status(
                        "Received model update request.",
                        log_level=fedn.Status.AUDIT,
                        type=fedn.StatusType.MODEL_UPDATE_REQUEST,
                        request=request,
                        sesssion_id=request.session_id,
                        sender_name=client_name
                    )

                    logger.info(f"Received task request of type {request.type} for model_id {request.model_id}")

                    callback(request)

        except grpc.RpcError as e:
            return self._handle_grpc_error(e, "TaskStream", lambda: self.listen_to_task_stream(client_name, client_id, callback))
        except Exception as e:
            logger.error(f"GRPC (TaskStream): An error occurred: {e}")
            self._disconnect()

    def send_status(self, msg: str, log_level=fedn.Status.INFO, type=None, request=None, sesssion_id: str = None, sender_name: str = None):
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
        status.sender.name = sender_name
        status.sender.role = fedn.WORKER
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
            self._disconnect()

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
            return self._handle_grpc_error(e, "Download", lambda: self.get_model_from_combiner(id, client_name, timeout))
        except Exception as e:
            logger.error(f"GRPC (Download): An error occurred: {e}")
            self._disconnect()

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
            self._disconnect()

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
            logger.info("Sending model update to combiner.")
            _ = self.combinerStub.SendModelUpdate(update, metadata=self.metadata)
        except grpc.RpcError as e:
            return self._handle_grpc_error(
                e,
                "SendModelUpdate",
                lambda: self.send_model_update(
                    sender_name,
                    sender_role,
                    model_id,
                    model_update_id,
                    receiver_name,
                    receiver_role,
                    meta
                )
            )
        except Exception as e:
            logger.error(f"GRPC (SendModelUpdate): An error occurred: {e}")
            self._disconnect()

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
            logger.info("Sending model validation to combiner.")
            _ = self.combinerStub.SendModelValidation(validation, metadata=self.metadata)
        except grpc.RpcError as e:
            return self._handle_grpc_error(
                e,
                "SendModelValidation",
                lambda: self.send_model_validation(
                    sender_name,
                    receiver_name,
                    receiver_role,
                    model_id,
                    metrics,
                    correlation_id,
                    session_id
                )
            )
        except Exception as e:
            logger.error(f"GRPC (SendModelValidation): An error occurred: {e}")
            self._disconnect()

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
        if status_code == grpc.StatusCode.UNAUTHENTICATED:
            details = e.details()
            if details == "Token expired":
                logger.warning(f"GRPC ({method_name}): Token expired.")
        #TODO: test this...
        self._disconnect()
        logger.error(f"GRPC ({method_name}): An error occurred: {e}")
        sys.exit("An error occurred during GRPC communication. Exiting.")

    def _disconnect(self):
        """Disconnect from the combiner."""
        self.channel.close()
        logger.info("Client {} disconnected.".format(self.name))
