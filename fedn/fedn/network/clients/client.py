import base64
import io
import json
import os
import queue
import re
import socket
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from distutils.dir_util import copy_tree
from io import BytesIO

import grpc
from cryptography.hazmat.primitives.serialization import Encoding
from google.protobuf.json_format import MessageToJson
from OpenSSL import SSL

import fedn.network.grpc.fedn_pb2 as fedn
import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.common.log_config import (logger, set_log_level_from_string,
                                    set_log_stream)
from fedn.network.clients.connect import ConnectorClient, Status
from fedn.network.clients.package import PackageRuntime
from fedn.network.clients.state import ClientState, ClientStateToString
from fedn.network.combiner.modelservice import upload_request_generator
from fedn.utils.dispatcher import Dispatcher
from fedn.utils.helpers.helpers import get_helper

CHUNK_SIZE = 1024 * 1024
VALID_NAME_REGEX = '^[a-zA-Z0-9_-]*$'


class GrpcAuth(grpc.AuthMetadataPlugin):
    def __init__(self, key):
        self._key = key

    def __call__(self, context, callback):
        callback((('authorization', f'Token {self._key}'),), None)


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
        self._attached = False
        self._missed_heartbeat = 0
        self.config = config

        set_log_level_from_string(config.get('verbosity', "INFO"))
        set_log_stream(config.get('logfile', None))

        self.connector = ConnectorClient(host=config['discover_host'],
                                         port=config['discover_port'],
                                         token=config['token'],
                                         name=config['name'],
                                         remote_package=config['remote_compute_context'],
                                         force_ssl=config['force_ssl'],
                                         verify=config['verify'],
                                         combiner=config['preferred_combiner'],
                                         id=config['client_id'])

        # Validate client name
        match = re.search(VALID_NAME_REGEX, config['name'])
        if not match:
            raise ValueError('Unallowed character in client name. Allowed characters: a-z, A-Z, 0-9, _, -.')

        self.name = config['name']
        dirname = time.strftime("%Y%m%d-%H%M%S")
        self.run_path = os.path.join(os.getcwd(), dirname)
        os.mkdir(self.run_path)

        self.started_at = datetime.now()
        self.logs = []

        self.inbox = queue.Queue()

        # Attach to the FEDn network (get combiner)
        client_config = self._attach()

        self._initialize_dispatcher(config)

        self._initialize_helper(client_config)
        if not self.helper:
            logger.warning("Failed to retrieve helper class settings: {}".format(
                client_config))

        self._subscribe_to_combiner(config)

        self.state = ClientState.idle

    def _assign(self):
        """Contacts the controller and asks for combiner assignment.

        :return: A configuration dictionary containing connection information for combiner.
        :rtype: dict
        """

        logger.info("Initiating assignment request.")
        while True:
            status, response = self.connector.assign()
            if status == Status.TryAgain:
                logger.info(response)
                time.sleep(5)
                continue
            if status == Status.Assigned:
                client_config = response
                break
            if status == Status.UnAuthorized:
                logger.critical(response)
                sys.exit("Exiting: Unauthorized")
            if status == Status.UnMatchedConfig:
                logger.critical(response)
                sys.exit("Exiting: UnMatchedConfig")
            time.sleep(5)

        logger.info("Assignment successfully received.")
        logger.info("Received combiner configuration: {}".format(client_config))
        return client_config

    def _add_grpc_metadata(self, key, value):
        """Add metadata for gRPC calls.

        :param key: The key of the metadata.
        :type key: str
        :param value: The value of the metadata.
        :type value: str
        """
        # Check if metadata exists and add if not
        if not hasattr(self, 'metadata'):
            self.metadata = ()

        # Check if metadata key already exists and replace value if so
        for i, (k, v) in enumerate(self.metadata):
            if k == key:
                # Replace value
                self.metadata = self.metadata[:i] + ((key, value),) + self.metadata[i + 1:]
                return

        # Set metadata using tuple concatenation
        self.metadata += ((key, value),)

    def _get_ssl_certificate(self, domain, port=443):
        context = SSL.Context(SSL.SSLv23_METHOD)
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

    def _connect(self, client_config):
        """Connect to assigned combiner.

        :param client_config: A configuration dictionary containing connection information for
        the combiner.
        :type client_config: dict
        """

        # TODO use the client_config['certificate'] for setting up secure comms'
        host = client_config['host']
        # Add host to gRPC metadata
        self._add_grpc_metadata('grpc-server', host)
        logger.info("Client using metadata: {}.".format(self.metadata))
        port = client_config['port']
        secure = False
        if client_config['fqdn'] is not None:
            host = client_config['fqdn']
            # assuming https if fqdn is used
            port = 443
        logger.info(f"Initiating connection to combiner host at: {host}:{port}")

        if client_config['certificate']:
            logger.info("Utilizing CA certificate for GRPC channel authentication.")
            secure = True
            cert = base64.b64decode(
                client_config['certificate'])  # .decode('utf-8')
            credentials = grpc.ssl_channel_credentials(root_certificates=cert)
            channel = grpc.secure_channel("{}:{}".format(host, str(port)), credentials)
        elif os.getenv("FEDN_GRPC_ROOT_CERT_PATH"):
            secure = True
            logger.info("Using root certificate from environment variable for GRPC channel.")
            with open(os.environ["FEDN_GRPC_ROOT_CERT_PATH"], 'rb') as f:
                credentials = grpc.ssl_channel_credentials(f.read())
            channel = grpc.secure_channel("{}:{}".format(host, str(port)), credentials)
        elif self.config['secure']:
            secure = True
            logger.info("Using CA certificate for GRPC channel.")
            cert = self._get_ssl_certificate(host, port=port)

            credentials = grpc.ssl_channel_credentials(cert.encode('utf-8'))
            if self.config['token']:
                token = self.config['token']
                auth_creds = grpc.metadata_call_credentials(GrpcAuth(token))
                channel = grpc.secure_channel("{}:{}".format(host, str(port)), grpc.composite_channel_credentials(credentials, auth_creds))
            else:
                channel = grpc.secure_channel("{}:{}".format(host, str(port)), credentials)
        else:
            logger.info("Using insecure GRPC channel.")
            if port == 443:
                port = 80
            channel = grpc.insecure_channel("{}:{}".format(
                host,
                str(port)))

        self.channel = channel

        self.connectorStub = rpc.ConnectorStub(channel)
        self.combinerStub = rpc.CombinerStub(channel)
        self.modelStub = rpc.ModelServiceStub(channel)

        logger.info("Successfully established {} connection to {}:{}".format("secure" if secure else "insecure",
                                                                             host,
                                                                             port))

        logger.info("Using {} compute package.".format(
            client_config["package"]))

    def _disconnect(self):
        """Disconnect from the combiner."""
        self.channel.close()

    def detach(self):
        """Detach from the FEDn network (disconnect from combiner)"""
        # Setting _attached to False will make all processing threads return
        if not self._attached:
            logger.info("Client is not attached.")

        self._attached = False
        # Close gRPC connection to combiner
        self._disconnect()

    def _attach(self):
        """Attach to the FEDn network (connect to combiner)"""
        # Ask controller for a combiner and connect to that combiner.
        if self._attached:
            logger.info("Client is already attached. ")
            return None

        client_config = self._assign()
        self._connect(client_config)

        if client_config:
            self._attached = True
        return client_config

    def _initialize_helper(self, client_config):
        """Initialize the helper class for the client.

        :param client_config: A configuration dictionary containing connection information for
        | the discovery service (controller) and settings governing e.g.
        | client-combiner assignment behavior.
        :type client_config: dict
        :return:
        """

        if 'helper_type' in client_config.keys():
            self.helper = get_helper(client_config['helper_type'])

    def _subscribe_to_combiner(self, config):
        """Listen to combiner message stream and start all processing threads.

        :param config: A configuration dictionary containing connection information for
        | the discovery service (controller) and settings governing e.g.
        | client-combiner assignment behavior.
        """

        # Start sending heartbeats to the combiner.
        threading.Thread(target=self._send_heartbeat, kwargs={
            'update_frequency': config['heartbeat_interval']}, daemon=True).start()

        # Start listening for combiner training and validation messages
        threading.Thread(
            target=self._listen_to_task_stream, daemon=True).start()
        self._attached = True

        # Start processing the client message inbox
        threading.Thread(target=self.process_request, daemon=True).start()

    def _initialize_dispatcher(self, config):
        """ Initialize the dispatcher for the client.

        :param config: A configuration dictionary containing connection information for
        | the discovery service (controller) and settings governing e.g.
        | client-combiner assignment behavior.
        :type config: dict
        :return:
        """
        if config['remote_compute_context']:
            pr = PackageRuntime(os.getcwd(), os.getcwd())

            retval = None
            tries = 10

            while tries > 0:
                retval = pr.download(
                    host=config['discover_host'],
                    port=config['discover_port'],
                    token=config['token'],
                    force_ssl=config['force_ssl'],
                    secure=config['secure']
                )
                if retval:
                    break
                time.sleep(60)
                logger.warning("Compute package not available. Retrying in 60 seconds. {} attempts remaining.".format(tries))
                tries -= 1

            if retval:
                if 'checksum' not in config:
                    logger.warning("Bypassing validation of package checksum. Ensure the package source is trusted.")
                else:
                    checks_out = pr.validate(config['checksum'])
                    if not checks_out:
                        logger.critical("Validation of local package failed. Client terminating.")
                        self.error_state = True
                        return

            if retval:
                pr.unpack()

            self.dispatcher = pr.dispatcher(self.run_path)
            try:
                logger.info("Initiating Dispatcher with entrypoint set to: startup")
                self.dispatcher.run_cmd("startup")
            except KeyError:
                pass
            except Exception as e:
                logger.error(f"Caught exception: {type(e).__name__}")
        else:
            # TODO: Deprecate
            dispatch_config = {'entry_points':
                               {'predict': {'command': 'python3 predict.py'},
                                'train': {'command': 'python3 train.py'},
                                'validate': {'command': 'python3 validate.py'}}}
            from_path = os.path.join(os.getcwd(), 'client')

            copy_tree(from_path, self.run_path)
            self.dispatcher = Dispatcher(dispatch_config, self.run_path)

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

        result = self.modelStub.Upload(upload_request_generator(bt, id), metadata=self.metadata)

        return result

    def _listen_to_task_stream(self):
        """Subscribe to the model update request stream.

        :return: None
        :rtype: None
        """

        r = fedn.ClientAvailableMessage()
        r.sender.name = self.name
        r.sender.role = fedn.WORKER
        # Add client to metadata
        self._add_grpc_metadata('client', self.name)

        while self._attached:
            try:
                for request in self.combinerStub.TaskStream(r, metadata=self.metadata):
                    if request:
                        logger.debug("Received model update request from combiner: {}.".format(request))
                    if request.sender.role == fedn.COMBINER:
                        # Process training request
                        self._send_status("Received model update request.", log_level=fedn.Status.AUDIT,
                                          type=fedn.StatusType.MODEL_UPDATE_REQUEST, request=request)
                        logger.info("Received model update request of type {} for model_id {}".format(request.type, request.model_id))

                        if request.type == fedn.StatusType.MODEL_UPDATE and self.config['trainer']:
                            self.inbox.put(('train', request))
                        elif request.type == fedn.StatusType.MODEL_VALIDATION and self.config['validator']:
                            self.inbox.put(('validate', request))
                        else:
                            logger.error("Unknown request type: {}".format(request.type))

            except grpc.RpcError as e:
                # Handle gRPC errors
                status_code = e.code()
                if status_code == grpc.StatusCode.UNAVAILABLE:
                    logger.warning("GRPC server unavailable during model update request stream. Retrying.")
                    # Retry after a delay
                    time.sleep(5)
                else:
                    # Log the error and continue
                    logger.error(f"An error occurred during model update request stream: {e}")

            except Exception as ex:
                # Handle other exceptions
                logger.error(f"An error occurred during model update request stream: {ex}")

        # Detach if not attached
        if not self._attached:
            return

    def _process_training_request(self, model_id):
        """Process a training (model update) request.

        :param model_id: The model id of the model to be updated.
        :type model_id: str
        :return: The model id of the updated model, or None if the update failed. And a dict with metadata.
        :rtype: tuple
        """

        self._send_status(
            "\t Starting processing of training request for model_id {}".format(model_id))
        self.state = ClientState.training

        try:
            meta = {}
            tic = time.time()
            mdl = self.get_model_from_combiner(str(model_id))
            if mdl is None:
                logger.error("Could not retrieve model from combiner. Aborting training request.")
                return None, None
            meta['fetch_model'] = time.time() - tic

            inpath = self.helper.get_tmp_path()
            with open(inpath, 'wb') as fh:
                fh.write(mdl.getbuffer())

            outpath = self.helper.get_tmp_path()
            tic = time.time()
            # TODO: Check return status, fail gracefully

            self.dispatcher.run_cmd("train {} {}".format(inpath, outpath))

            meta['exec_training'] = time.time() - tic

            tic = time.time()
            out_model = None

            with open(outpath, "rb") as fr:
                out_model = io.BytesIO(fr.read())

            # Stream model update to combiner server
            updated_model_id = uuid.uuid4()
            self.send_model_to_combiner(out_model, str(updated_model_id))
            meta['upload_model'] = time.time() - tic

            # Read the metadata file
            with open(outpath+'-metadata', 'r') as fh:
                training_metadata = json.loads(fh.read())
            meta['training_metadata'] = training_metadata

            os.unlink(inpath)
            os.unlink(outpath)
            os.unlink(outpath+'-metadata')

        except Exception as e:
            logger.error("Could not process training request due to error: {}".format(e))
            updated_model_id = None
            meta = {'status': 'failed', 'error': str(e)}

        self.state = ClientState.idle

        return updated_model_id, meta

    def _process_validation_request(self, model_id, is_inference):
        """Process a validation request.

        :param model_id: The model id of the model to be validated.
        :type model_id: str
        :param is_inference: True if the validation is an inference request, False if it is a validation request.
        :type is_inference: bool
        :return: The validation metrics, or None if validation failed.
        :rtype: dict
        """
        # Figure out cmd
        if is_inference:
            cmd = 'infer'
        else:
            cmd = 'validate'

        self._send_status(
            f"Processing {cmd} request for model_id {model_id}")
        self.state = ClientState.validating
        try:
            model = self.get_model_from_combiner(str(model_id))
            if model is None:
                logger.error("Could not retrieve model from combiner. Aborting validation request.")
                return None
            inpath = self.helper.get_tmp_path()

            with open(inpath, "wb") as fh:
                fh.write(model.getbuffer())

            _, outpath = tempfile.mkstemp()
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

    def process_request(self):
        """Process training and validation tasks. """
        while True:

            if not self._attached:
                return

            try:
                (task_type, request) = self.inbox.get(timeout=1.0)
                if task_type == 'train':

                    tic = time.time()
                    self.state = ClientState.training
                    model_id, meta = self._process_training_request(
                        request.model_id)
                    processing_time = time.time()-tic
                    meta['processing_time'] = processing_time
                    meta['config'] = request.data

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
                        # TODO: Check responses
                        _ = self.combinerStub.SendModelUpdate(update, metadata=self.metadata)
                        self._send_status("Model update completed.", log_level=fedn.Status.AUDIT,
                                          type=fedn.StatusType.MODEL_UPDATE, request=update)

                    else:
                        self._send_status("Client {} failed to complete model update.",
                                          log_level=fedn.Status.WARNING,
                                          request=request)
                    self.state = ClientState.idle
                    self.inbox.task_done()

                elif task_type == 'validate':
                    self.state = ClientState.validating
                    metrics = self._process_validation_request(
                        request.model_id, False)

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

                        _ = self.combinerStub.SendModelValidation(
                            validation, metadata=self.metadata)

                        status_type = fedn.StatusType.MODEL_VALIDATION

                        self._send_status("Model validation completed.", log_level=fedn.Status.AUDIT,
                                          type=status_type, request=validation)
                    else:
                        self._send_status("Client {} failed to complete model validation.".format(self.name),
                                          log_level=fedn.Status.WARNING, request=request)

                    self.state = ClientState.idle
                    self.inbox.task_done()
            except queue.Empty:
                pass

    def _handle_combiner_failure(self):
        """ Register failed combiner connection."""
        self._missed_heartbeat += 1
        if self._missed_heartbeat > self.config['reconnect_after_missed_heartbeat']:
            self.detach()()

    def _send_heartbeat(self, update_frequency=2.0):
        """Send a heartbeat to the combiner.

        :param update_frequency: The frequency of the heartbeat in seconds.
        :type update_frequency: float
        :return: None if the client is detached.
        :rtype: None
        """
        while True:
            heartbeat = fedn.Heartbeat(sender=fedn.Client(
                name=self.name, role=fedn.WORKER))
            try:
                self.connectorStub.SendHeartbeat(heartbeat, metadata=self.metadata)
                self._missed_heartbeat = 0
            except grpc.RpcError as e:
                status_code = e.code()
                logger.warning("Client heartbeat: GRPC error, {}. Retrying.".format(
                    status_code.name))
                logger.debug(e)
                self._handle_combiner_failure()

            time.sleep(update_frequency)
            if not self._attached:
                return

    def _send_status(self, msg, log_level=fedn.Status.INFO, type=None, request=None):
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
        status.sender.name = self.name
        status.sender.role = fedn.WORKER
        status.log_level = log_level
        status.status = str(msg)
        if type is not None:
            status.type = type

        if request is not None:
            status.data = MessageToJson(request)

        self.logs.append(
            "{} {} LOG LEVEL {} MESSAGE {}".format(str(datetime.now()), status.sender.name, status.log_level,
                                                   status.status))
        _ = self.connectorStub.SendStatus(status, metadata=self.metadata)

    def run(self):
        """ Run the client. """
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
                if not self._attached:
                    logger.info("Detached from combiner.")
                    # TODO: Implement a check/condition to ulitmately close down if too many reattachment attepts have failed. s
                    self._attach()
                    self._subscribe_to_combiner(self.config)
                if self.error_state:
                    return
        except KeyboardInterrupt:
            logger.info("Shutting down.")
