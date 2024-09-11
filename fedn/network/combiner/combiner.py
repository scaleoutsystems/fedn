import json
import queue
import re
import signal
import threading
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import TypedDict

import fedn.network.grpc.fedn_pb2 as fedn
import fedn.network.grpc.fedn_pb2_grpc as rpc
from fedn.common.certificate.certificate import Certificate
from fedn.common.log_config import logger, set_log_level_from_string, set_log_stream
from fedn.network.combiner.roundhandler import RoundConfig, RoundHandler
from fedn.network.combiner.shared import repository, statestore
from fedn.network.grpc.server import Server, ServerConfig

VALID_NAME_REGEX = "^[a-zA-Z0-9_-]*$"


class Role(Enum):
    """Enum for combiner roles."""

    WORKER = 1
    COMBINER = 2
    REDUCER = 3
    OTHER = 4


def role_to_proto_role(role):
    """Convert a Role to a proto Role.

    :param role: the role to convert
    :type role: :class:`fedn.network.combiner.server.Role`
    :return: proto role
    :rtype: :class:`fedn.network.grpc.fedn_pb2.Role`
    """
    if role == Role.COMBINER:
        return fedn.COMBINER
    if role == Role.WORKER:
        return fedn.WORKER
    if role == Role.REDUCER:
        return fedn.REDUCER
    if role == Role.OTHER:
        return fedn.OTHER


class CombinerConfig(TypedDict):
    """Configuration for the combiner."""

    discover_host: str
    discover_port: int
    token: str
    host: str
    port: int
    ip: str
    parent: str
    fqdn: str
    name: str
    secure: bool
    verify: bool
    cert_path: str
    key_path: str
    max_clients: int
    network_id: str
    logfile: str
    verbosity: str


class Combiner(rpc.CombinerServicer, rpc.ReducerServicer, rpc.ConnectorServicer, rpc.ControlServicer):
    """Combiner gRPC server.

    :param config: configuration for the combiner
    :type config: dict
    """

    def __init__(self, config):
        """Initialize Combiner server."""
        set_log_level_from_string(config.get("verbosity", "INFO"))
        set_log_stream(config.get("logfile", None))

        # Client queues
        self.clients = {}

        # Validate combiner name
        match = re.search(VALID_NAME_REGEX, config["name"])
        if not match:
            raise ValueError("Unallowed character in combiner name. Allowed characters: a-z, A-Z, 0-9, _, -.")

        self.id = config["name"]
        self.role = Role.COMBINER
        self.max_clients = config["max_clients"]

        # Set up model repository
        self.repository = repository

        self.statestore = statestore

        # Add combiner to statestore
        interface_config = {
            "port": config["port"],
            "fqdn": config["fqdn"],
            "name": config["name"],
            "address": config["host"],
            "parent": "localhost",
            "ip": "",
        }
        self.statestore.set_combiner(interface_config)

        # Fetch all clients previously connected to the combiner
        # If a client and a combiner goes down at the same time,
        # the client will be stuck listed as "online" in the statestore.
        # Set the status to offline for previous clients.
        previous_clients = self.statestore.clients.find({"combiner": config["name"]})
        for client in previous_clients:
            try:
                self.statestore.set_client({"name": client["name"], "status": "offline", "client_id": client["client_id"]})
            except KeyError:
                self.statestore.set_client({"name": client["name"], "status": "offline"})

        # Set up gRPC server configuration
        if config["secure"]:
            cert = Certificate(key_path=config["key_path"], cert_path=config["cert_path"])
            certificate, key = cert.get_keypair_raw()
            grpc_server_config = ServerConfig(port=config["port"], secure=True, key=key, certificate=certificate)
        else:
            grpc_server_config = ServerConfig(port=config["port"], secure=False)

        # Create gRPC server
        self.server = Server(self, grpc_server_config)

        # Set up round controller
        self.round_handler = RoundHandler(self)

        # Start thread for round controller
        threading.Thread(target=self.round_handler.run, daemon=True).start()

        # Start thread for client status updates: TODO: Should be configurable
        threading.Thread(target=self._deamon_thread_client_status, daemon=True).start()

        # Start the gRPC server
        self.server.start()

    def __whoami(self, client, instance):
        """Set the client id and role in a proto message.

        :param client: the client to set the id and role for
        :type client: :class:`fedn.network.grpc.fedn_pb2.Client`
        :param instance: the instance to get the id and role from
        :type instance: :class:`fedn.network.combiner.server.Combiner`
        :return: the client with id and role set
        :rtype: :class:`fedn.network.grpc.fedn_pb2.Client`
        """
        client.name = instance.id
        client.role = role_to_proto_role(instance.role)
        return client

    def request_model_update(self, session_id, model_id, config, clients=[]):
        """Ask clients to update the current global model.

        :param config: the model configuration to send to clients
        :type config: dict
        :param clients: the clients to send the request to
        :type clients: list

        """
        clients = self._send_request_type(fedn.StatusType.MODEL_UPDATE, session_id, model_id, config, clients)

        if len(clients) < 20:
            logger.info("Sent model update request for model {} to clients {}".format(model_id, clients))
        else:
            logger.info("Sent model update request for model {} to {} clients".format(model_id, len(clients)))

    def request_model_validation(self, session_id, model_id, clients=[]):
        """Ask clients to validate the current global model.

        :param model_id: the model id to validate
        :type model_id: str
        :param config: the model configuration to send to clients
        :type config: dict
        :param clients: the clients to send the request to
        :type clients: list

        """
        clients = self._send_request_type(fedn.StatusType.MODEL_VALIDATION, session_id, model_id, clients)

        if len(clients) < 20:
            logger.info("Sent model validation request for model {} to clients {}".format(model_id, clients))
        else:
            logger.info("Sent model validation request for model {} to {} clients".format(model_id, len(clients)))

    def request_model_inference(self, session_id: str, model_id: str, clients: list = []) -> None:
        """Ask clients to perform inference on the model.

        :param model_id: the model id to perform inference on
        :type model_id: str
        :param config: the model configuration to send to clients
        :type config: dict
        :param clients: the clients to send the request to
        :type clients: list

        """
        clients = self._send_request_type(fedn.StatusType.INFERENCE, session_id, model_id, clients)

        if len(clients) < 20:
            logger.info("Sent model inference request for model {} to clients {}".format(model_id, clients))
        else:
            logger.info("Sent model inference request for model {} to {} clients".format(model_id, len(clients)))

    def _send_request_type(self, request_type, session_id, model_id, config=None, clients=[]):
        """Send a request of a specific type to clients.

        :param request_type: the type of request
        :type request_type: :class:`fedn.network.grpc.fedn_pb2.StatusType`
        :param model_id: the model id to send in the request
        :type model_id: str
        :param config: the model configuration to send to clients
        :type config: dict
        :param clients: the clients to send the request to
        :type clients: list
        :return: the clients
        :rtype: list
        """
        if len(clients) == 0:
            if request_type == fedn.StatusType.MODEL_UPDATE:
                clients = self.get_active_trainers()
            elif request_type == fedn.StatusType.MODEL_VALIDATION:
                clients = self.get_active_validators()
            elif request_type == fedn.StatusType.INFERENCE:
                # TODO: add inference clients type
                clients = self.get_active_validators()
        for client in clients:
            request = fedn.TaskRequest()
            request.model_id = model_id
            request.correlation_id = str(uuid.uuid4())
            request.timestamp = str(datetime.now())
            request.type = request_type
            request.session_id = session_id

            request.sender.name = self.id
            request.sender.role = fedn.COMBINER
            request.receiver.client_id = client
            request.receiver.role = fedn.WORKER
            # Set the request data, not used in validation
            if request_type == fedn.StatusType.INFERENCE:
                presigned_url = self.repository.presigned_put_url(self.repository.inference_bucket, f"{client}/{session_id}")
                # TODO: in inference, request.data should also contain user-defined data/parameters
                request.data = json.dumps({"presigned_url": presigned_url})
            elif request_type == fedn.StatusType.MODEL_UPDATE:
                request.data = json.dumps(config)
            self._put_request_to_client_queue(request, fedn.Queue.TASK_QUEUE)
        return clients

    def get_active_trainers(self):
        """Get a list of active trainers.

        :return: the list of active trainers
        :rtype: list
        """
        trainers = self._list_active_clients(fedn.Queue.TASK_QUEUE)
        return trainers

    def get_active_validators(self):
        """Get a list of active validators.

        :return: the list of active validators
        :rtype: list
        """
        validators = self._list_active_clients(fedn.Queue.TASK_QUEUE)
        return validators

    def nr_active_trainers(self):
        """Get the number of active trainers.

        :return: the number of active trainers
        :rtype: int
        """
        return len(self.get_active_trainers())

    ####################################################################################################################

    def __join_client(self, client):
        """Add a client to the list of active clients.

        :param client: the client to add
        :type client: :class:`fedn.network.grpc.fedn_pb2.Client`
        """
        if client.client_id not in self.clients.keys():
            # The status is set to offline by default, and will be updated once _list_active_clients is called.
            self.clients[client.client_id] = {"last_seen": datetime.now(), "status": "offline"}

    def _subscribe_client_to_queue(self, client, queue_name):
        """Subscribe a client to the queue.

        :param client: the client to subscribe
        :type client: :class:`fedn.network.grpc.fedn_pb2.Client`
        :param queue_name: the name of the queue to subscribe to
        :type queue_name: str
        """
        self.__join_client(client)
        if queue_name not in self.clients[client.client_id].keys():
            self.clients[client.client_id][queue_name] = queue.Queue()

    def __get_queue(self, client, queue_name):
        """Get the queue for a client.

        :param client: the client to get the queue for
        :type client: :class:`fedn.network.grpc.fedn_pb2.Client`
        :param queue_name: the name of the queue to get
        :type queue_name: str
        :return: the queue
        :rtype: :class:`queue.Queue`

        :raises KeyError: if the queue does not exist
        """
        try:
            return self.clients[client.client_id][queue_name]
        except KeyError:
            raise

    def _list_subscribed_clients(self, queue_name):
        """List all clients subscribed to a queue.

        :param queue_name: the name of the queue
        :type queue_name: str
        :return: a list of client names
        :rtype: list
        """
        subscribed_clients = []
        for name, client in self.clients.items():
            if queue_name in client.keys():
                subscribed_clients.append(name)
        return subscribed_clients

    def _list_active_clients(self, channel):
        """List all clients that have sent a status message in the last 10 seconds.

        :param channel: the name of the channel
        :type channel: str
        :return: a list of client names
        :rtype: list
        """
        # Temporary dict to store client status
        clients = {
            "active_clients": [],
            "update_active_clients": [],
            "update_offline_clients": [],
        }
        for client in self._list_subscribed_clients(channel):
            status = self.clients[client]["status"]
            now = datetime.now()
            then = self.clients[client]["last_seen"]
            if (now - then) < timedelta(seconds=10):
                clients["active_clients"].append(client)
                # If client has changed status, update statestore
                if status != "online":
                    self.clients[client]["status"] = "online"
                    clients["update_active_clients"].append(client)
            elif status != "offline":
                self.clients[client]["status"] = "offline"
                clients["update_offline_clients"].append(client)
        # Update statestore with client status
        if len(clients["update_active_clients"]) > 0:
            self.statestore.update_client_status(clients["update_active_clients"], "online")
        if len(clients["update_offline_clients"]) > 0:
            self.statestore.update_client_status(clients["update_offline_clients"], "offline")

        return clients["active_clients"]

    def _deamon_thread_client_status(self, timeout=5):
        """Deamon thread that checks for inactive clients and updates statestore."""
        while True:
            time.sleep(timeout)
            # TODO: Also update validation clients
            self._list_active_clients(fedn.Queue.TASK_QUEUE)

    def _put_request_to_client_queue(self, request, queue_name):
        """Get a client specific queue and add a request to it.
        The client is identified by the request.receiver.

        :param request: the request to send
        :type request: :class:`fedn.network.grpc.fedn_pb2.Request`
        :param queue_name: the name of the queue to send the request to
        :type queue_name: str
        """
        try:
            q = self.__get_queue(request.receiver, queue_name)
            q.put(request)
        except Exception as e:
            logger.error("Failed to put request to client queue {} for client {}: {}".format(queue_name, request.receiver.name, str(e)))
            raise

    def _send_status(self, status):
        """Report a status to backend db.

        :param status: the status to report
        :type status: :class:`fedn.network.grpc.fedn_pb2.Status`
        """
        self.statestore.report_status(status)

    def _flush_model_update_queue(self):
        """Clear the model update queue (aggregator).

        :return: True if successful, else False
        """
        q = self.round_handler.aggregator.model_updates
        try:
            with q.mutex:
                q.queue.clear()
                q.all_tasks_done.notify_all()
                q.unfinished_tasks = 0
            return True
        except Exception as e:
            logger.error("Failed to flush model update queue: %s", str(e))
            return False

    #####################################################################################################################

    # Controller Service

    def Start(self, control: fedn.ControlRequest, context):
        """Start a round of federated learning"

        :param control: the control request
        :type control: :class:`fedn.network.grpc.fedn_pb2.ControlRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the control response
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ControlResponse`
        """
        logger.info("grpc.Combiner.Start: Starting round")

        config = RoundConfig()
        for parameter in control.parameter:
            config.update({parameter.key: parameter.value})

        logger.debug("grpc.Combiner.Start: Round config {}".format(config))

        job_id = self.round_handler.push_round_config(config)
        logger.info("grcp.Combiner.Start: Pushed round config (job_id): {}".format(job_id))

        response = fedn.ControlResponse()
        p = response.parameter.add()
        p.key = "job_id"
        p.value = job_id

        return response

    def SetAggregator(self, control: fedn.ControlRequest, context):
        """Set the active aggregator.

        :param control: the control request
        :type control: :class:`fedn.network.grpc.fedn_pb2.ControlRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the control response
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ControlResponse`
        """
        logger.debug("grpc.Combiner.SetAggregator: Called")
        for parameter in control.parameter:
            aggregator = parameter.value

        status = self.round_handler.set_aggregator(aggregator)

        response = fedn.ControlResponse()
        if status:
            response.message = "Success"
        else:
            response.message = "Failed"

        return response

    def FlushAggregationQueue(self, control: fedn.ControlRequest, context):
        """Flush the queue.

        :param control: the control request
        :type control: :class:`fedn.network.grpc.fedn_pb2.ControlRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the control response
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ControlResponse`
        """
        logger.debug("grpc.Combiner.FlushAggregationQueue: Called")
        status = self._flush_model_update_queue()

        response = fedn.ControlResponse()
        if status:
            response.message = "Success"
        else:
            response.message = "Failed"

        return response

    ##############################################################################

    def Stop(self, control: fedn.ControlRequest, context):
        """TODO: Not yet implemented.

        :param control: the control request
        :type control: :class:`fedn.network.grpc.fedn_pb2.ControlRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the control response
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ControlResponse`
        """
        response = fedn.ControlResponse()
        logger.info("grpc.Combiner.Stop: Called")
        return response

    #####################################################################################################################

    def SendStatus(self, status: fedn.Status, context):
        """A client RPC endpoint that accepts status messages.

        :param status: the status message
        :type status: :class:`fedn.network.grpc.fedn_pb2.Status`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the response
        :rtype: :class:`fedn.network.grpc.fedn_pb2.Response`
        """
        logger.debug("grpc.Combiner.SendStatus: Called")
        self._send_status(status)

        response = fedn.Response()
        response.response = "Status received."
        return response

    def ListActiveClients(self, request: fedn.ListClientsRequest, context):
        """RPC endpoint that returns a ClientList containing the names of all active clients.
            An active client has sent a status message / responded to a heartbeat
            request in the last 10 seconds.

        :param request: the request
        :type request: :class:`fedn.network.grpc.fedn_pb2.ListClientsRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the client list
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ClientList`
        """
        clients = fedn.ClientList()
        active_clients = self._list_active_clients(request.channel)
        nr_active_clients = len(active_clients)
        if nr_active_clients < 20:
            logger.info("grpc.Combiner.ListActiveClients:  Active clients: {}".format(active_clients))
        else:
            logger.info("grpc.Combiner.ListActiveClients: Number active clients: {}".format(nr_active_clients))

        for client in active_clients:
            clients.client.append(fedn.Client(name=client, role=fedn.WORKER))
        return clients

    def AcceptingClients(self, request: fedn.ConnectionRequest, context):
        """RPC endpoint that returns a ConnectionResponse indicating whether the server
        is accepting clients or not.

        :param request: the request (unused)
        :type request: :class:`fedn.network.grpc.fedn_pb2.ConnectionRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the response
        :rtype: :class:`fedn.network.grpc.fedn_pb2.ConnectionResponse`
        """
        response = fedn.ConnectionResponse()
        active_clients = self._list_active_clients(fedn.Queue.TASK_QUEUE)

        try:
            requested = int(self.max_clients)
            if len(active_clients) >= requested:
                response.status = fedn.ConnectionStatus.NOT_ACCEPTING
                return response
            if len(active_clients) < requested:
                response.status = fedn.ConnectionStatus.ACCEPTING
                return response

        except Exception as e:
            logger.error("Combiner not properly configured! {}".format(e))
            raise

        response.status = fedn.ConnectionStatus.TRY_AGAIN_LATER
        return response

    def SendHeartbeat(self, heartbeat: fedn.Heartbeat, context):
        """RPC that lets clients send a hearbeat, notifying the server that
            the client is available.

        :param heartbeat: the heartbeat
        :type heartbeat: :class:`fedn.network.grpc.fedn_pb2.Heartbeat`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the response
        :rtype: :class:`fedn.network.grpc.fedn_pb2.Response`
        """
        logger.debug("GRPC: Received heartbeat from {}".format(heartbeat.sender.name))
        # Update the clients dict with the last seen timestamp.
        client = heartbeat.sender
        self.__join_client(client)
        self.clients[client.client_id]["last_seen"] = datetime.now()

        response = fedn.Response()
        response.sender.name = heartbeat.sender.name
        response.sender.role = heartbeat.sender.role
        response.response = "Heartbeat received"
        return response

    # Combiner Service

    def TaskStream(self, response, context):
        """A server stream RPC endpoint (Update model). Messages from client stream.

        :param response: the response
        :type response: :class:`fedn.network.grpc.fedn_pb2.ModelUpdateRequest`
        :param context: the context
        :type context: :class:`grpc._server._Context`
        """
        client = response.sender
        metadata = context.invocation_metadata()
        if metadata:
            metadata = dict(metadata)
            logger.info("grpc.Combiner.TaskStream: Client connected: {}\n".format(metadata["client"]))

        status = fedn.Status(status="Client {} connecting to TaskStream.".format(client.name))
        status.log_level = fedn.Status.INFO
        status.timestamp.GetCurrentTime()

        self.__whoami(status.sender, self)

        self._subscribe_client_to_queue(client, fedn.Queue.TASK_QUEUE)
        q = self.__get_queue(client, fedn.Queue.TASK_QUEUE)

        self._send_status(status)

        # Set client status to online
        self.clients[client.client_id]["status"] = "online"
        self.statestore.set_client({"name": client.name, "status": "online", "client_id": client.client_id, "last_seen": datetime.now()})

        # Keep track of the time context has been active
        start_time = time.time()
        while context.is_active():
            # Check if the context has been active for more than 10 seconds
            if time.time() - start_time > 10:
                self.clients[client.client_id]["last_seen"] = datetime.now()
                # Reset the start time
                start_time = time.time()
            try:
                yield q.get(timeout=1.0)
            except queue.Empty:
                pass
            except Exception as e:
                logger.error("Error in ModelUpdateRequestStream: {}".format(e))

    def SendModelUpdate(self, request, context):
        """Send a model update response.

        :param request: the request
        :type request: :class:`fedn.network.grpc.fedn_pb2.ModelUpdate`
        :param context: the context
        :type context: :class:`grpc._server._Context`
        :return: the response
        :rtype: :class:`fedn.network.grpc.fedn_pb2.Response`
        """
        self.round_handler.aggregator.on_model_update(request)

        response = fedn.Response()
        response.response = "RECEIVED ModelUpdate {} from client  {}".format(response, response.sender.name)
        return response  # TODO Fill later

    def register_model_validation(self, validation):
        """Register a model validation.

        :param validation: the model validation
        :type validation: :class:`fedn.network.grpc.fedn_pb2.ModelValidation`
        """
        self.statestore.report_validation(validation)

    def SendModelValidation(self, request, context):
        """Send a model validation response.

        :param request: the request
        :type request: :class:`fedn.network.grpc.fedn_pb2.ModelValidation`
        :param context: the context
        :type context: :class:`grpc._server._Context`
        :return: the response
        :rtype: :class:`fedn.network.grpc.fedn_pb2.Response`
        """
        logger.info("Recieved ModelValidation from {}".format(request.sender.name))

        self.register_model_validation(request)

        response = fedn.Response()
        response.response = "RECEIVED ModelValidation {} from client  {}".format(response, response.sender.name)
        return response

    ####################################################################################################################

    def run(self):
        """Start the server."""
        logger.info("COMBINER: {} started, ready for gRPC requests.".format(self.id))
        try:
            while True:
                signal.pause()
        except (KeyboardInterrupt, SystemExit):
            pass
        self.server.stop()
