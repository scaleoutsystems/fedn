import io
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from io import BytesIO

import grpc
import requests
from google.protobuf.json_format import MessageToJson
from tenacity import retry, stop_after_attempt

import fedn.network.grpc.fedn_pb2 as fedn
import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.common.config import FEDN_AUTH_SCHEME, FEDN_PACKAGE_EXTRACT_DIR
from fedn.common.log_config import logger, set_log_level_from_string, set_log_stream
from fedn.network.clients.connect import ConnectorClient, Status
from fedn.network.clients.package import PackageRuntime
from fedn.network.clients.state import ClientState, ClientStateToString
from fedn.network.combiner.modelservice import get_tmp_path, upload_request_generator
from fedn.utils.helpers.helpers import get_helper

CHUNK_SIZE = 1024 * 1024
VALID_NAME_REGEX = "^[a-zA-Z0-9_-]*$"


class GrpcAuth(grpc.AuthMetadataPlugin):
    def __init__(self, key):
        self._key = key

    def __call__(self, context, callback):
        callback((("authorization", f"{FEDN_AUTH_SCHEME} {self._key}"),), None)


class Client:
    """FEDn Client. Service running on client/datanodes in a federation,
    recieving and handling model update and model validation requests.

    :param config: A configuration dictionary containing connection information for the discovery service (controller)
        and settings governing e.g. client-combiner assignment behavior.
    :type config: dict
    """

    def __init__(self, config):
        """Initialize the client."""
        self.state = None
        self.error_state = False
        self._connected = False
        self._missed_heartbeat = 0
        self.config = config
        self.trace_attribs = False
        set_log_level_from_string(config.get("verbosity", "INFO"))
        set_log_stream(config.get("logfile", None))

        self.id = config["client_id"] or str(uuid.uuid4())

        # Validate client name
        match = re.search(VALID_NAME_REGEX, config["name"])
        if not match:
            raise ValueError("Unallowed character in client name. Allowed characters: a-z, A-Z, 0-9, _, -.")

        # Folder where the client will store downloaded compute package and logs
        self.name = config["name"]
        if FEDN_PACKAGE_EXTRACT_DIR:
            self.run_path = os.path.join(os.getcwd(), FEDN_PACKAGE_EXTRACT_DIR)
        else:
            dirname = self.name + "-" + time.strftime("%Y%m%d-%H%M%S")
            self.run_path = os.path.join(os.getcwd(), dirname)
        if not os.path.exists(self.run_path):
            os.mkdir(self.run_path)

        self.started_at = datetime.now()
        self.logs = []

        self.inbox = queue.Queue()

        # Attach to the FEDn network (get combiner or attach directly)
        if config["combiner"]:
            combiner_config = {"status": "assigned", "host": config["combiner"], "port": config["combiner_port"], "helper_type": ""}
            if config["proxy_server"]:
                combiner_config["fqdn"] = config["proxy_server"]
        else:
            self.connector = ConnectorClient(
                host=config["discover_host"],
                port=config["discover_port"],
                token=config["token"],
                name=config["name"],
                remote_package=config["remote_compute_context"],
                force_ssl=config["force_ssl"],
                verify=config["verify"],
                combiner=config["preferred_combiner"],
                id=self.id,
            )
            combiner_config = self.assign()
        self.connect(combiner_config)

        self._initialize_dispatcher(self.config)

        self._initialize_helper(combiner_config)
        if not self.helper:
            logger.warning("Failed to retrieve helper class settings: {}".format(combiner_config))

        self._subscribe_to_combiner(self.config)

        self.state = ClientState.idle

    def assign(self):
        """Contacts the controller and asks for combiner assignment.

        :return: A configuration dictionary containing connection information for combiner.
        :rtype: dict
        """
        logger.info("Initiating assignment request.")
        while True:
            status, response = self.connector.assign()
            if status == Status.TryAgain:
                logger.warning(response)
                logger.info("Assignment request failed. Retrying in 5 seconds.")
                time.sleep(5)
                continue
            if status == Status.Assigned:
                combiner_config = response
                break
            if status == Status.UnAuthorized:
                logger.critical(response)
                sys.exit("Exiting: Unauthorized")
            if status == Status.UnMatchedConfig:
                logger.critical(response)
                sys.exit("Exiting: UnMatchedConfig")
            time.sleep(5)
        logger.info("Assignment successfully received.")
        logger.info("Received combiner configuration: {}".format(combiner_config))
        return combiner_config

    def _add_grpc_metadata(self, key, value):
        """Add metadata for gRPC calls.

        :param key: The key of the metadata.
        :type key: str
        :param value: The value of the metadata.
        :type value: str
        """
        # Check if metadata exists and add if not
        if not hasattr(self, "metadata"):
            self.metadata = ()

        # Check if metadata key already exists and replace value if so
        for i, (k, v) in enumerate(self.metadata):
            if k == key:
                # Replace value
                self.metadata = self.metadata[:i] + ((key, value),) + self.metadata[i + 1 :]
                return

        # Set metadata using tuple concatenation
        self.metadata += ((key, value),)

    def connect(self, combiner_config):
        """Connect to combiner.

        :param combiner_config: connection information for the combiner.
        :type combiner_config: dict
        """
        if self._connected:
            logger.info("Client is already attached. ")
            return

        # TODO use the combiner_config['certificate'] for setting up secure comms'
        host = combiner_config["host"]
        # Add host to gRPC metadata
        self._add_grpc_metadata("grpc-server", host)
        logger.debug("Client using metadata: {}.".format(self.metadata))
        port = combiner_config["port"]
        secure = False
        if combiner_config["fqdn"] is not None:
            host = combiner_config["fqdn"]
            # assuming https if fqdn is used
            port = 443
        logger.info(f"Initiating connection to combiner host at: {host}:{port}")

        if os.getenv("FEDN_GRPC_ROOT_CERT_PATH"):
            secure = True
            logger.info("Using root certificate from environment variable for GRPC channel.")
            with open(os.environ["FEDN_GRPC_ROOT_CERT_PATH"], "rb") as f:
                credentials = grpc.ssl_channel_credentials(f.read())
            channel = grpc.secure_channel("{}:{}".format(host, str(port)), credentials)
        elif self.config["secure"]:
            secure = True
            logger.info("Using default location for root certificates.")
            credentials = grpc.ssl_channel_credentials()
            if self.config["token"]:
                token = self.config["token"]
                auth_creds = grpc.metadata_call_credentials(GrpcAuth(token))
                channel = grpc.secure_channel("{}:{}".format(host, str(port)), grpc.composite_channel_credentials(credentials, auth_creds))
            else:
                channel = grpc.secure_channel("{}:{}".format(host, str(port)), credentials)
        else:
            logger.info("Using insecure GRPC channel.")
            if port == 443:
                port = 80
            channel = grpc.insecure_channel("{}:{}".format(host, str(port)))

        self.channel = channel

        self.connectorStub = rpc.ConnectorStub(channel)
        self.combinerStub = rpc.CombinerStub(channel)
        self.modelStub = rpc.ModelServiceStub(channel)

        logger.info("Successfully established {} connection to {}:{}".format("secure" if secure else "insecure", host, port))

        self._connected = True

    def disconnect(self):
        """Disconnect from the combiner."""
        if not self._connected:
            logger.info("Client is not connected.")

        self.channel.close()
        self._connected = False
        logger.info("Client {} disconnected.".format(self.name))

    def _initialize_helper(self, combiner_config):
        """Initialize the helper class for the client.

        :param combiner_config: A configuration dictionary containing connection information for
        | the discovery service (controller) and settings governing e.g.
        | client-combiner assignment behavior.
        :type combiner_config: dict
        :return:
        """
        if "helper_type" in combiner_config.keys():
            if not combiner_config["helper_type"]:
                # Default to numpyhelper
                self.helper = get_helper("numpyhelper")
            else:
                self.helper = get_helper(combiner_config["helper_type"])

    def _subscribe_to_combiner(self, config):
        """Listen to combiner message stream and start all processing threads.

        :param config: A configuration dictionary containing connection information for
        | the discovery service (controller) and settings governing e.g.
        | client-combiner assignment behavior.
        """
        # Start sending heartbeats to the combiner.
        threading.Thread(target=self._send_heartbeat, kwargs={"update_frequency": config["heartbeat_interval"]}, daemon=True).start()

        # Start listening for combiner training and validation messages
        threading.Thread(target=self._listen_to_task_stream, daemon=True).start()
        self._connected = True

        # Start processing the client message inbox
        threading.Thread(target=self.process_request, daemon=True).start()

    @retry(stop=stop_after_attempt(3))
    def untar_package(self, package_runtime):
        _, package_runpath = package_runtime.unpack()
        return package_runpath

    def _initialize_dispatcher(self, config):
        """Initialize the dispatcher for the client.

        :param config: A configuration dictionary containing connection information for
        | the discovery service (controller) and settings governing e.g.
        | client-combiner assignment behavior.
        :type config: dict
        :return:
        """
        pr = PackageRuntime(self.run_path)
        if config["remote_compute_context"]:
            retval = None
            tries = 10

            while tries > 0:
                retval = pr.download(
                    host=config["discover_host"], port=config["discover_port"], token=config["token"], force_ssl=config["force_ssl"], secure=config["secure"]
                )
                if retval:
                    break
                time.sleep(60)
                logger.warning("Compute package not available. Retrying in 60 seconds. {} attempts remaining.".format(tries))
                tries -= 1

            if retval:
                if "checksum" not in config:
                    logger.warning("Bypassing validation of package checksum. Ensure the package source is trusted.")
                else:
                    checks_out = pr.validate(config["checksum"])
                    if not checks_out:
                        logger.critical("Validation of local package failed. Client terminating.")
                        self.error_state = True
                        return
            package_runpath = ""
            if retval:
                package_runpath = self.untar_package(pr)

            self.dispatcher = pr.dispatcher(package_runpath)
            try:
                logger.info("Initiating Dispatcher with entrypoint set to: startup")
                activate_cmd = self.dispatcher._get_or_create_python_env()
                self.dispatcher.run_cmd("startup")
            except KeyError:
                logger.info("No startup command found in package. Continuing.")
                pass
            except Exception as e:
                logger.error(f"Caught exception: {type(e).__name__}")

        else:
            from_path = os.path.join(os.getcwd(), "client")
            self.dispatcher = pr.dispatcher(from_path)
        # Get or create python environment
        activate_cmd = self.dispatcher._get_or_create_python_env()
        if activate_cmd:
            logger.info("To activate the virtual environment, run: {}".format(activate_cmd))

    def get_model_from_combiner(self, id, timeout=20):
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
        request.sender.name = self.name
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

    def send_model_to_combiner(self, model, id):
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

    def _listen_to_task_stream(self):
        """Subscribe to the model update request stream.

        :return: None
        :rtype: None
        """
        r = fedn.ClientAvailableMessage()
        r.sender.name = self.name
        r.sender.role = fedn.WORKER
        r.sender.client_id = self.id
        # Add client to metadata
        self._add_grpc_metadata("client", self.name)
        status_code = None

        while self._connected:
            try:
                if status_code == grpc.StatusCode.UNAVAILABLE:
                    logger.info("GRPC TaskStream: server available again.")
                    status_code = None
                for request in self.combinerStub.TaskStream(r, metadata=self.metadata):
                    if request:
                        logger.debug("Received model update request from combiner: {}.".format(request))
                    if request.sender.role == fedn.COMBINER:
                        # Process training request
                        self.send_status(
                            "Received model update request.",
                            log_level=fedn.Status.AUDIT,
                            type=fedn.StatusType.MODEL_UPDATE_REQUEST,
                            request=request,
                            sesssion_id=request.session_id,
                        )
                        logger.info("Received task request of type {} for model_id {}".format(request.type, request.model_id))

                        if request.type == fedn.StatusType.MODEL_UPDATE and self.config["trainer"]:
                            self.inbox.put(("train", request))
                        elif request.type == fedn.StatusType.MODEL_VALIDATION and self.config["validator"]:
                            self.inbox.put(("validate", request))
                        elif request.type == fedn.StatusType.INFERENCE and self.config["validator"]:
                            logger.info("Received inference request for model_id {}".format(request.model_id))
                            presigned_url = json.loads(request.data)
                            presigned_url = presigned_url["presigned_url"]
                            logger.info("Inference presigned URL: {}".format(presigned_url))
                            self.inbox.put(("infer", request))
                        else:
                            logger.error("Unknown request type: {}".format(request.type))

            except grpc.RpcError as e:
                # Handle gRPC errors
                status_code = e.code()
                if status_code == grpc.StatusCode.UNAVAILABLE:
                    logger.warning("GRPC TaskStream: server unavailable during model update request stream. Retrying.")
                    # Retry after a delay
                    time.sleep(5)
                    continue
                if status_code == grpc.StatusCode.UNAUTHENTICATED:
                    details = e.details()
                    if details == "Token expired":
                        logger.warning("GRPC TaskStream: Token expired. Reconnecting.")
                        self.disconnect()

                if status_code == grpc.StatusCode.CANCELLED:
                    # Expected if the client is disconnected
                    logger.critical("GRPC TaskStream: Client disconnected from combiner. Trying to reconnect.")

                else:
                    # Log the error and continue
                    logger.error(f"GRPC TaskStream: An error occurred during model update request stream: {e}")

            except Exception as ex:
                # Handle other exceptions
                logger.error(f"GRPC TaskStream: An error occurred during model update request stream: {ex}")

        # Detach if not attached
        if not self._connected:
            return

    def _process_training_request(self, model_id: str, session_id: str = None):
        """Process a training (model update) request.

        :param model_id: The model id of the model to be updated.
        :type model_id: str
        :param session_id: The id of the current session
        :type session_id: str
        :return: The model id of the updated model, or None if the update failed. And a dict with metadata.
        :rtype: tuple
        """
        self.send_status("\t Starting processing of training request for model_id {}".format(model_id), sesssion_id=session_id)
        self.state = ClientState.training

        try:
            meta = {}
            tic = time.time()
            mdl = self.get_model_from_combiner(str(model_id))
            if mdl is None:
                logger.error("Could not retrieve model from combiner. Aborting training request.")
                return None, None
            meta["fetch_model"] = time.time() - tic

            inpath = self.helper.get_tmp_path()
            with open(inpath, "wb") as fh:
                fh.write(mdl.getbuffer())

            outpath = self.helper.get_tmp_path()
            tic = time.time()
            # TODO: Check return status, fail gracefully

            self.dispatcher.run_cmd("train {} {}".format(inpath, outpath))

            meta["exec_training"] = time.time() - tic

            tic = time.time()
            out_model = None

            with open(outpath, "rb") as fr:
                out_model = io.BytesIO(fr.read())

            # Stream model update to combiner server
            updated_model_id = uuid.uuid4()
            self.send_model_to_combiner(out_model, str(updated_model_id))
            meta["upload_model"] = time.time() - tic

            # Read the metadata file
            with open(outpath + "-metadata", "r") as fh:
                training_metadata = json.loads(fh.read())
            meta["training_metadata"] = training_metadata

            os.unlink(inpath)
            os.unlink(outpath)
            os.unlink(outpath + "-metadata")

        except Exception as e:
            logger.error("Could not process training request due to error: {}".format(e))
            updated_model_id = None
            meta = {"status": "failed", "error": str(e)}

        self.state = ClientState.idle

        return updated_model_id, meta

    def _process_validation_request(self, model_id: str, is_inference: bool, session_id: str = None):
        """Process a validation request.

        :param model_id: The model id of the model to be validated.
        :type model_id: str
        :param is_inference: True if the validation is an inference request, False if it is a validation request.
        :type is_inference: bool
        :param session_id: The id of the current session.
        :type session_id: str
        :return: The validation metrics, or None if validation failed.
        :rtype: dict
        """
        # Figure out cmd
        if is_inference:
            cmd = "infer"
        else:
            cmd = "validate"

        self.send_status(f"Processing {cmd} request for model_id {model_id}", sesssion_id=session_id)
        self.state = ClientState.validating
        try:
            model = self.get_model_from_combiner(str(model_id))
            if model is None:
                logger.error("Could not retrieve model from combiner. Aborting validation request.")
                return None
            inpath = self.helper.get_tmp_path()

            with open(inpath, "wb") as fh:
                fh.write(model.getbuffer())

            outpath = get_tmp_path()
            self.dispatcher.run_cmd(f"{cmd} {inpath} {outpath}")

            with open(outpath, "r") as fh:
                validation = json.loads(fh.read())

            os.unlink(inpath)
            os.unlink(outpath)

        except Exception as e:
            logger.warning("Validation failed with exception {}".format(e))
            self.state = ClientState.idle
            return None

        self.state = ClientState.idle
        return validation

    def _process_inference_request(self, model_id: str, session_id: str, presigned_url: str):
        """Process an inference request.

        :param model_id: The model id of the model to be used for inference.
        :type model_id: str
        :param session_id: The id of the current session.
        :type session_id: str
        :param presigned_url: The presigned URL for the data to be used for inference.
        :type presigned_url: str
        :return: None
        """
        self.send_status(f"Processing inference request for model_id {model_id}", sesssion_id=session_id)
        try:
            model = self.get_model_from_combiner(str(model_id))
            if model is None:
                logger.error("Could not retrieve model from combiner. Aborting inference request.")
                return
            inpath = self.helper.get_tmp_path()

            with open(inpath, "wb") as fh:
                fh.write(model.getbuffer())

            outpath = get_tmp_path()
            self.dispatcher.run_cmd(f"predict {inpath} {outpath}")

            # Upload the inference result to the presigned URL
            with open(outpath, "rb") as fh:
                response = requests.put(presigned_url, data=fh.read())

            os.unlink(inpath)
            os.unlink(outpath)

            if response.status_code != 200:
                logger.warning("Inference upload failed with status code {}".format(response.status_code))
                self.state = ClientState.idle
                return

        except Exception as e:
            logger.warning("Inference failed with exception {}".format(e))
            self.state = ClientState.idle
            return

        self.state = ClientState.idle
        return

    def process_request(self):
        """Process training and validation tasks."""
        while True:
            if not self._connected:
                return

            try:
                (task_type, request) = self.inbox.get(timeout=1.0)
                if task_type == "train":
                    tic = time.time()
                    self.state = ClientState.training
                    model_id, meta = self._process_training_request(request.model_id, session_id=request.session_id)

                    if meta is not None:
                        processing_time = time.time() - tic
                        meta["processing_time"] = processing_time
                        meta["config"] = request.data

                    if model_id is not None:
                        # Send model update to combiner
                        update = fedn.ModelUpdate()
                        update.sender.name = self.name
                        update.sender.role = fedn.WORKER
                        update.receiver.name = request.sender.name
                        update.receiver.role = request.sender.role
                        update.model_id = request.model_id
                        update.model_update_id = str(model_id)
                        update.timestamp = str(datetime.now())
                        update.correlation_id = request.correlation_id
                        update.meta = json.dumps(meta)

                        try:
                            _ = self.combinerStub.SendModelUpdate(update, metadata=self.metadata)
                            self.send_status(
                                "Model update completed.",
                                log_level=fedn.Status.AUDIT,
                                type=fedn.StatusType.MODEL_UPDATE,
                                request=update,
                                sesssion_id=request.session_id,
                            )
                        except grpc.RpcError as e:
                            status_code = e.code()
                            logger.error("GRPC error, {}.".format(status_code.name))
                            logger.debug(e)
                        except ValueError as e:
                            logger.error("GRPC error, RPC channel closed. {}".format(e))
                            logger.debug(e)
                    else:
                        self.send_status(
                            "Client {} failed to complete model update.", log_level=fedn.Status.WARNING, request=request, sesssion_id=request.session_id
                        )

                    self.state = ClientState.idle
                    self.inbox.task_done()

                elif task_type == "validate":
                    self.state = ClientState.validating
                    metrics = self._process_validation_request(request.model_id, False, request.session_id)

                    if metrics is not None:
                        # Send validation
                        validation = fedn.ModelValidation()
                        validation.sender.name = self.name
                        validation.sender.role = fedn.WORKER
                        validation.receiver.name = request.sender.name
                        validation.receiver.role = request.sender.role
                        validation.model_id = str(request.model_id)
                        validation.data = json.dumps(metrics)
                        validation.timestamp.GetCurrentTime()
                        validation.correlation_id = request.correlation_id
                        validation.session_id = request.session_id

                        try:
                            _ = self.combinerStub.SendModelValidation(validation, metadata=self.metadata)

                            status_type = fedn.StatusType.MODEL_VALIDATION
                            self.send_status(
                                "Model validation completed.", log_level=fedn.Status.AUDIT, type=status_type, request=validation, sesssion_id=request.session_id
                            )
                        except grpc.RpcError as e:
                            status_code = e.code()
                            logger.error("GRPC error, {}.".format(status_code.name))
                            logger.debug(e)
                        except ValueError as e:
                            logger.error("GRPC error, RPC channel closed. {}".format(e))
                            logger.debug(e)
                    else:
                        self.send_status(
                            "Client {} failed to complete model validation.".format(self.name),
                            log_level=fedn.Status.WARNING,
                            request=request,
                            sesssion_id=request.session_id,
                        )

                    self.state = ClientState.idle
                    self.inbox.task_done()
                elif task_type == "infer":
                    self.state = ClientState.inferencing
                    try:
                        presigned_url = json.loads(request.data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode inference request data: {e}")
                        self.state = ClientState.idle
                        continue

                    if "presigned_url" not in presigned_url:
                        logger.error("Inference request missing presigned_url.")
                        self.state = ClientState.idle
                        continue
                    presigned_url = presigned_url["presigned_url"]
                    _ = self._process_inference_request(request.model_id, request.session_id, presigned_url)
                    self.state = ClientState.idle
            except queue.Empty:
                pass
            except grpc.RpcError as e:
                logger.critical(f"GRPC process_request: An error occurred during process request: {e}")

    def _send_heartbeat(self, update_frequency=2.0):
        """Send a heartbeat to the combiner.

        :param update_frequency: The frequency of the heartbeat in seconds.
        :type update_frequency: float
        :return: None if the client is disconnected.
        :rtype: None
        """
        while True:
            heartbeat = fedn.Heartbeat(sender=fedn.Client(name=self.name, role=fedn.WORKER, client_id=self.id))
            try:
                self.connectorStub.SendHeartbeat(heartbeat, metadata=self.metadata)
                if self._missed_heartbeat > 0:
                    logger.info("GRPC heartbeat: combiner available again after {} missed heartbeats.".format(self._missed_heartbeat))
                self._missed_heartbeat = 0
            except grpc.RpcError as e:
                status_code = e.code()
                if status_code == grpc.StatusCode.UNAVAILABLE:
                    self._missed_heartbeat += 1
                    logger.error(
                        "GRPC hearbeat: combiner unavailable, retrying (attempt {}/{}).".format(
                            self._missed_heartbeat, self.config["reconnect_after_missed_heartbeat"]
                        )
                    )
                    if self._missed_heartbeat > self.config["reconnect_after_missed_heartbeat"]:
                        self.disconnect()
                        self._missed_heartbeat = 0
                if status_code == grpc.StatusCode.UNAUTHENTICATED:
                    details = e.details()
                    if details == "Token expired":
                        logger.error("GRPC hearbeat: Token expired. Disconnecting.")
                        self.disconnect()
                        sys.exit("Unauthorized. Token expired. Please obtain a new token.")
                logger.debug(e)

            time.sleep(update_frequency)
            if not self._connected:
                logger.info("SendStatus: Client disconnected.")
                return

    def send_status(self, msg, log_level=fedn.Status.INFO, type=None, request=None, sesssion_id: str = None):
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
        if not self._connected:
            logger.info("SendStatus: Client disconnected.")
            return

        status = fedn.Status()
        status.timestamp.GetCurrentTime()
        status.sender.name = self.name
        status.sender.role = fedn.WORKER
        status.log_level = log_level
        status.status = str(msg)
        status.session_id = sesssion_id
        if type is not None:
            status.type = type

        if request is not None:
            status.data = MessageToJson(request)

        self.logs.append("{} {} LOG LEVEL {} MESSAGE {}".format(str(datetime.now()), status.sender.name, status.log_level, status.status))
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

    def run(self):
        """Run the client."""
        try:
            cnt = 0
            old_state = self.state
            while True:
                time.sleep(1)
                if cnt == 0:
                    logger.info("Client is active, waiting for model update requests.")
                    cnt = 1
                if self.state != old_state:
                    logger.info("Client in {} state.".format(ClientStateToString(self.state)))
                if not self._connected:
                    logger.warning("Client lost connection to combiner. Attempting to reconnect to FEDn network.")
                    combiner_config = self.assign()
                    self.connect(combiner_config)
                    self._subscribe_to_combiner(self.config)
                    cnt = 0
                if self.error_state:
                    logger.error("Client in error state. Terminiating.")
                    sys.exit("Client in error state. Terminiating.")
        except KeyboardInterrupt:
            logger.info("Shutting down.")
