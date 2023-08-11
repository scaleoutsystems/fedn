import base64
import json
import queue
import re
import signal
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum

import fedn.common.net.grpc.fedn_pb2 as fedn
import fedn.common.net.grpc.fedn_pb2_grpc as rpc
from fedn.common.net.grpc.server import Server
from fedn.common.storage.s3.s3repo import S3ModelRepository
from fedn.common.tracer.mongotracer import MongoTracer
from fedn.network.combiner.connect import ConnectorCombiner, Status
from fedn.network.combiner.modelservice import ModelService
from fedn.network.combiner.round import RoundController

VALID_NAME_REGEX = '^[a-zA-Z0-9_-]*$'


class Role(Enum):
    """ Enum for combiner roles. """
    WORKER = 1
    COMBINER = 2
    REDUCER = 3
    OTHER = 4


def role_to_proto_role(role):
    """ Convert a Role to a proto Role.

    :param role: the role to convert
    :type role: :class:`fedn.network.combiner.server.Role`
    :return: proto role
    :rtype: :class:`fedn.common.net.grpc.fedn_pb2.Role`
    """
    if role == Role.COMBINER:
        return fedn.COMBINER
    if role == Role.WORKER:
        return fedn.WORKER
    if role == Role.REDUCER:
        return fedn.REDUCER
    if role == Role.OTHER:
        return fedn.OTHER


class Combiner(rpc.CombinerServicer, rpc.ReducerServicer, rpc.ConnectorServicer, rpc.ControlServicer):
    """ Combiner gRPC server.

    :param config: configuration for the combiner
    :type config: dict
    """

    def __init__(self, config):
        """ Initialize Combiner server."""

        # Client queues
        self.clients = {}

        self.modelservice = ModelService()

        # Validate combiner name
        match = re.search(VALID_NAME_REGEX, config['name'])
        if not match:
            raise ValueError('Unallowed character in combiner name. Allowed characters: a-z, A-Z, 0-9, _, -.')

        self.id = config['name']
        self.role = Role.COMBINER
        self.max_clients = config['max_clients']

        # Connector to announce combiner to discover service (reducer)
        announce_client = ConnectorCombiner(host=config['discover_host'],
                                            port=config['discover_port'],
                                            myhost=config['host'],
                                            fqdn=config['fqdn'],
                                            myport=config['port'],
                                            token=config['token'],
                                            name=config['name'],
                                            secure=config['secure'],
                                            verify=config['verify'])

        response = None
        while True:
            # announce combiner to discover service
            status, response = announce_client.announce()
            if status == Status.TryAgain:
                print(response, flush=True)
                time.sleep(5)
                continue
            if status == Status.Assigned:
                announce_config = response
                print(
                    "COMBINER {0}: Announced successfully".format(self.id), flush=True)
                break
            if status == Status.UnAuthorized:
                print(response, flush=True)
                sys.exit("Exiting: Unauthorized")
            if status == Status.UnMatchedConfig:
                print(response, flush=True)
                sys.exit("Exiting: Missing config")

        cert = announce_config['certificate']
        key = announce_config['key']

        if announce_config['certificate']:
            cert = base64.b64decode(announce_config['certificate'])  # .decode('utf-8')
            key = base64.b64decode(announce_config['key'])  # .decode('utf-8')

        # Set up gRPC server configuration
        grpc_config = {'port': config['port'],
                       'secure': config['secure'],
                       'certificate': cert,
                       'key': key}

        # Set up model repository
        self.repository = S3ModelRepository(
            announce_config['storage']['storage_config'])

        # Create gRPC server
        self.server = Server(self, self.modelservice, grpc_config)

        # Set up tracer for statestore
        self.tracer = MongoTracer(
            announce_config['statestore']['mongo_config'], announce_config['statestore']['network_id'])

        # Set up round controller
        self.control = RoundController(config['aggregator'], self.repository, self, self.modelservice)

        # Start thread for round controller
        threading.Thread(target=self.control.run, daemon=True).start()

        # Start the gRPC server
        self.server.start()

    def __whoami(self, client, instance):
        """ Set the client id and role in a proto message.

        :param client: the client to set the id and role for
        :type client: :class:`fedn.common.net.grpc.fedn_pb2.Client`
        :param instance: the instance to get the id and role from
        :type instance: :class:`fedn.network.combiner.server.Combiner`
        :return: the client with id and role set
        :rtype: :class:`fedn.common.net.grpc.fedn_pb2.Client`
        """
        client.name = instance.id
        client.role = role_to_proto_role(instance.role)
        return client

    def report_status(self, msg, log_level=fedn.Status.INFO, type=None, request=None, flush=True):
        """ Report status of the combiner.

        :param msg: the message to report
        :type msg: str
        :param log_level: the log level to report at
        :type log_level: :class:`fedn.common.net.grpc.fedn_pb2.Status`
        :param type: the type of status to report
        :type type: :class:`fedn.common.net.grpc.fedn_pb2.Status.Type`
        :param request: the request to report status for
        :type request: :class:`fedn.common.net.grpc.fedn_pb2.Request`
        :param flush: whether to flush the message to stdout
        :type flush: bool
        """
        print("{}:COMBINER({}):{} {}".format(datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S'), self.id, log_level, msg), flush=flush)

    def request_model_update(self, config, clients=[]):
        """ Ask clients to update the current global model.

        :param config: the model configuration to send to clients
        :type config: dict
        :param clients: the clients to send the request to
        :type clients: list

        """

        request = fedn.ModelUpdateRequest()
        self.__whoami(request.sender, self)
        request.model_id = config['model_id']
        request.correlation_id = str(uuid.uuid4())
        request.timestamp = str(datetime.now())
        request.data = json.dumps(config)

        if len(clients) == 0:
            clients = self.get_active_trainers()

        for client in clients:
            request.receiver.name = client.name
            request.receiver.role = fedn.WORKER
            _ = self.SendModelUpdateRequest(request, self)
            # TODO: Check response

        print("COMBINER: Sent model update request for model {} to clients {}".format(
            request.model_id, clients), flush=True)

    def request_model_validation(self, model_id, config, clients=[]):
        """ Ask clients to validate the current global model.

        :param model_id: the model id to validate
        :type model_id: str
        :param config: the model configuration to send to clients
        :type config: dict
        :param clients: the clients to send the request to
        :type clients: list

        """

        request = fedn.ModelValidationRequest()
        self.__whoami(request.sender, self)
        request.model_id = model_id
        request.correlation_id = str(uuid.uuid4())
        request.timestamp = str(datetime.now())
        request.is_inference = (config['task'] == 'inference')

        if len(clients) == 0:
            clients = self.get_active_validators()

        for client in clients:
            request.receiver.name = client.name
            request.receiver.role = fedn.WORKER
            self.SendModelValidationRequest(request, self)

        print("COMBINER: Sent validation request for model {} to clients {}".format(
            model_id, clients), flush=True)

    def _list_clients(self, channel):
        """ List active clients on a channel.

        :param channel: the channel to list clients for, for example MODEL_UPDATE_REQUESTS
        :type channel: :class:`fedn.common.net.grpc.fedn_pb2.Channel`
        :return: the list of active clients
        :rtype: list
        """
        request = fedn.ListClientsRequest()
        self.__whoami(request.sender, self)
        request.channel = channel
        clients = self.ListActiveClients(request, self)
        return clients.client

    def get_active_trainers(self):
        """ Get a list of active trainers.

        :return: the list of active trainers
        :rtype: list
        """
        trainers = self._list_clients(fedn.Channel.MODEL_UPDATE_REQUESTS)
        return trainers

    def get_active_validators(self):
        """ Get a list of active validators.

        :return: the list of active validators
        :rtype: list
        """
        validators = self._list_clients(fedn.Channel.MODEL_VALIDATION_REQUESTS)
        return validators

    def nr_active_trainers(self):
        """ Get the number of active trainers.

        :return: the number of active trainers
        :rtype: int
        """
        return len(self.get_active_trainers())

    def nr_active_validators(self):
        """ Get the number of active validators.

        :return: the number of active validators
        :rtype: int
        """
        return len(self.get_active_validators())

    ####################################################################################################################

    def __join_client(self, client):
        """ Add a client to the list of active clients.

        :param client: the client to add
        :type client: :class:`fedn.common.net.grpc.fedn_pb2.Client`
        """
        if client.name not in self.clients.keys():
            self.clients[client.name] = {"lastseen": datetime.now()}

    def _subscribe_client_to_queue(self, client, queue_name):
        """ Subscribe a client to the queue.

        :param client: the client to subscribe
        :type client: :class:`fedn.common.net.grpc.fedn_pb2.Client`
        :param queue_name: the name of the queue to subscribe to
        :type queue_name: str
        """
        self.__join_client(client)
        if queue_name not in self.clients[client.name].keys():
            self.clients[client.name][queue_name] = queue.Queue()

    def __get_queue(self, client, queue_name):
        """ Get the queue for a client.

        :param client: the client to get the queue for
        :type client: :class:`fedn.common.net.grpc.fedn_pb2.Client`
        :param queue_name: the name of the queue to get
        :type queue_name: str
        :return: the queue
        :rtype: :class:`queue.Queue`

        :raises KeyError: if the queue does not exist
        """
        try:
            return self.clients[client.name][queue_name]
        except KeyError:
            raise

    def _send_request(self, request, queue_name):
        """ Send a request to a client.

        :param request: the request to send
        :type request: :class:`fedn.common.net.grpc.fedn_pb2.Request`
        :param queue_name: the name of the queue to send the request to
        :type queue_name: str
        """
        self.__route_request_to_client(request, request.receiver, queue_name)

    def _broadcast_request(self, request, queue_name):
        """ Publish a request to all subscribed members.

        :param request: the request to send
        :type request: :class:`fedn.common.net.grpc.fedn_pb2.Request`
        :param queue_name: the name of the queue to send the request to
        :type queue_name: str
        """
        active_clients = self._list_active_clients()
        for client in active_clients:
            self.clients[client.name][queue_name].put(request)

    def __route_request_to_client(self, request, client, queue_name):
        """ Route a request to a client.

        :param request: the request to send
        :type request: :class:`fedn.common.net.grpc.fedn_pb2.Request`
        :param client: the client to send the request to
        :type client: :class:`fedn.common.net.grpc.fedn_pb2.Client`
        :param queue_name: the name of the queue to send the request to
        :type queue_name: str

        :raises Exception: if the request could not be routed, direct cause of KeyError in __get_queue
        """
        try:
            q = self.__get_queue(client, queue_name)
            q.put(request)
        except Exception:
            print("Failed to route request to client: {} {}",
                  request.receiver, queue_name)
            raise

    def _send_status(self, status):
        """ Report a status to tracer.

        :param status: the status to report
        :type status: :class:`fedn.common.net.grpc.fedn_pb2.Status`
        """

        self.tracer.report_status(status)

    def __register_heartbeat(self, client):
        """ Register a client if first time connecting. Update heartbeat timestamp.

        :param client: the client to register
        :type client: :class:`fedn.common.net.grpc.fedn_pb2.Client`
        """
        self.__join_client(client)
        self.clients[client.name]["lastseen"] = datetime.now()

    #####################################################################################################################

    # Control Service

    def Start(self, control: fedn.ControlRequest, context):
        """ Start a round of federated learning"

        :param control: the control request
        :type control: :class:`fedn.common.net.grpc.fedn_pb2.ControlRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the control response
        :rtype: :class:`fedn.common.net.grpc.fedn_pb2.ControlResponse`
        """
        print("\nRECIEVED **START** from Controller {}\n".format(control.command), flush=True)

        config = {}
        for parameter in control.parameter:
            config.update({parameter.key: parameter.value})

        job_id = self.control.push_round_config(config)

        response = fedn.ControlResponse()
        p = response.parameter.add()
        p.key = "job_id"
        p.value = job_id

        return response

    def Configure(self, control: fedn.ControlRequest, context):
        """ Configure the Combiner.

        :param control: the control request
        :type control: :class:`fedn.common.net.grpc.fedn_pb2.ControlRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the control response
        :rtype: :class:`fedn.common.net.grpc.fedn_pb2.ControlResponse`
        """
        for parameter in control.parameter:
            setattr(self, parameter.key, parameter.value)

        response = fedn.ControlResponse()
        return response

    def Stop(self, control: fedn.ControlRequest, context):
        """ TODO: Not yet implemented.

        :param control: the control request
        :type control: :class:`fedn.common.net.grpc.fedn_pb2.ControlRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the control response
        :rtype: :class:`fedn.common.net.grpc.fedn_pb2.ControlResponse`
        """
        response = fedn.ControlResponse()
        print("\n RECIEVED **STOP** from Controller\n", flush=True)
        return response

    def Report(self, control: fedn.ControlRequest, context):
        """ Describe current state of the Combiner.

        :param control: the control request
        :type control: :class:`fedn.common.net.grpc.fedn_pb2.ControlRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the control response
        :rtype: :class:`fedn.common.net.grpc.fedn_pb2.ControlResponse`
        """

        response = fedn.ControlResponse()
        self.report_status("\n RECIEVED **REPORT** from Controller\n",
                           log_level=fedn.Status.INFO)

        control_state = self.control.aggregator.get_state()
        self.report_status("Aggregator state: {}".format(control_state), log_level=fedn.Status.INFO)
        p = response.parameter.add()
        for key, value in control_state.items():
            p.key = str(key)
            p.value = str(value)

        active_trainers = self.get_active_trainers()
        p = response.parameter.add()
        p.key = "nr_active_trainers"
        p.value = str(len(active_trainers))

        active_validators = self.get_active_validators()
        p = response.parameter.add()
        p.key = "nr_active_validators"
        p.value = str(len(active_validators))

        active_trainers_ = self.get_active_trainers()
        active_trainers = []
        for client in active_trainers_:
            active_trainers.append(client)
        p = response.parameter.add()
        p.key = "active_trainers"
        p.value = str(active_trainers)

        active_validators_ = self.get_active_validators()
        active_validators = []
        for client in active_validators_:
            active_validators.append(client)
        p = response.parameter.add()
        p.key = "active_validators"
        p.value = str(active_validators)

        p = response.parameter.add()
        p.key = "nr_active_clients"
        p.value = str(len(active_trainers)+len(active_validators))

        p = response.parameter.add()
        p.key = "nr_unprocessed_compute_plans"
        p.value = str(self.control.round_configs.qsize())

        p = response.parameter.add()
        p.key = "name"
        p.value = str(self.id)

        return response

    #####################################################################################################################

    def AllianceStatusStream(self, response, context):
        """ A server stream RPC endpoint that emits status messages.

        :param response: the response
        :type response: :class:`fedn.common.net.grpc.fedn_pb2.Response`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`"""
        status = fedn.Status(
            status="Client {} connecting to AllianceStatusStream.".format(response.sender))
        status.log_level = fedn.Status.INFO
        status.sender.name = self.id
        status.sender.role = role_to_proto_role(self.role)
        self._subscribe_client_to_queue(response.sender, fedn.Channel.STATUS)
        q = self.__get_queue(response.sender, fedn.Channel.STATUS)
        self._send_status(status)

        while True:
            yield q.get()

    def SendStatus(self, status: fedn.Status, context):
        """ A client stream RPC endpoint that accepts status messages.

        :param status: the status message
        :type status: :class:`fedn.common.net.grpc.fedn_pb2.Status`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the response
        :rtype: :class:`fedn.common.net.grpc.fedn_pb2.Response`
        """

        self._send_status(status)

        response = fedn.Response()
        response.response = "Status received."
        return response

    def _list_subscribed_clients(self, queue_name):
        """ List all clients subscribed to a queue.

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
        """ List all clients that have sent a status message in the last 10 seconds.

        :param channel: the name of the channel
        :type channel: str
        :return: a list of client names
        :rtype: list
        """
        active_clients = []
        for client in self._list_subscribed_clients(channel):
            # This can break with different timezones.
            now = datetime.now()
            then = self.clients[client]["lastseen"]
            # TODO: move the heartbeat timeout to config.
            if (now - then) < timedelta(seconds=10):
                active_clients.append(client)
        return active_clients

    def _drop_inactive_clients(self):
        """ TODO: Not implemented. Clean up clients that have missed the heartbeat. """

    def ListActiveClients(self, request: fedn.ListClientsRequest, context):
        """ RPC endpoint that returns a ClientList containing the names of all active clients.
            An active client has sent a status message / responded to a heartbeat
            request in the last 10 seconds.

        :param request: the request
        :type request: :class:`fedn.common.net.grpc.fedn_pb2.ListClientsRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the client list
        :rtype: :class:`fedn.common.net.grpc.fedn_pb2.ClientList`
        """
        clients = fedn.ClientList()
        active_clients = self._list_active_clients(request.channel)

        for client in active_clients:
            clients.client.append(fedn.Client(name=client, role=fedn.WORKER))
        return clients

    def AcceptingClients(self, request: fedn.ConnectionRequest, context):
        """ RPC endpoint that returns a ConnectionResponse indicating whether the server
        is accepting clients or not.

        :param request: the request (unused)
        :type request: :class:`fedn.common.net.grpc.fedn_pb2.ConnectionRequest`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the response
        :rtype: :class:`fedn.common.net.grpc.fedn_pb2.ConnectionResponse`
        """
        response = fedn.ConnectionResponse()
        active_clients = self._list_active_clients(
            fedn.Channel.MODEL_UPDATE_REQUESTS)

        try:
            requested = int(self.max_clients)
            if len(active_clients) >= requested:
                response.status = fedn.ConnectionStatus.NOT_ACCEPTING
                return response
            if len(active_clients) < requested:
                response.status = fedn.ConnectionStatus.ACCEPTING
                return response

        except Exception as e:
            print("Combiner not properly configured! {}".format(e), flush=True)
            raise

        response.status = fedn.ConnectionStatus.TRY_AGAIN_LATER
        return response

    def SendHeartbeat(self, heartbeat: fedn.Heartbeat, context):
        """ RPC that lets clients send a hearbeat, notifying the server that
            the client is available.

        :param heartbeat: the heartbeat
        :type heartbeat: :class:`fedn.common.net.grpc.fedn_pb2.Heartbeat`
        :param context: the context (unused)
        :type context: :class:`grpc._server._Context`
        :return: the response
        :rtype: :class:`fedn.common.net.grpc.fedn_pb2.Response`
        """
        self.__register_heartbeat(heartbeat.sender)
        response = fedn.Response()
        response.sender.name = heartbeat.sender.name
        response.sender.role = heartbeat.sender.role
        response.response = "Heartbeat received"
        return response

    # Combiner Service

    def ModelUpdateStream(self, update, context):
        """ Model update stream RPC endpoint. Update status for client is connecting to stream.

        :param update: the update message
        :type update: :class:`fedn.common.net.grpc.fedn_pb2.ModelUpdate`
        :param context: the context
        :type context: :class:`grpc._server._Context`
        """
        client = update.sender
        status = fedn.Status(
            status="Client {} connecting to ModelUpdateStream.".format(client.name))
        status.log_level = fedn.Status.INFO
        status.sender.name = self.id
        status.sender.role = role_to_proto_role(self.role)

        self._subscribe_client_to_queue(client, fedn.Channel.MODEL_UPDATES)
        q = self.__get_queue(client, fedn.Channel.MODEL_UPDATES)

        self._send_status(status)

        while context.is_active():
            try:
                yield q.get(timeout=1.0)
            except queue.Empty:
                pass

    def ModelUpdateRequestStream(self, response, context):
        """ A server stream RPC endpoint (Update model). Messages from client stream.

        :param response: the response
        :type response: :class:`fedn.common.net.grpc.fedn_pb2.ModelUpdateRequest`
        :param context: the context
        :type context: :class:`grpc._server._Context`
        """

        client = response.sender
        metadata = context.invocation_metadata()
        if metadata:
            metadata = dict(metadata)
            print("\nClient connected: {}\n".format(metadata['client']), flush=True)

        status = fedn.Status(
            status="Client {} connecting to ModelUpdateRequestStream.".format(client.name))
        status.log_level = fedn.Status.INFO
        status.timestamp = str(datetime.now())

        self.__whoami(status.sender, self)

        self._subscribe_client_to_queue(
            client, fedn.Channel.MODEL_UPDATE_REQUESTS)
        q = self.__get_queue(client, fedn.Channel.MODEL_UPDATE_REQUESTS)

        self._send_status(status)

        while context.is_active():
            try:
                yield q.get(timeout=1.0)
            except queue.Empty:
                pass

    def ModelValidationStream(self, update, context):
        """ Model validation stream RPC endpoint. Update status for client is connecting to stream.

        :param update: the update message
        :type update: :class:`fedn.common.net.grpc.fedn_pb2.ModelValidation`
        :param context: the context
        :type context: :class:`grpc._server._Context`
        """
        client = update.sender
        status = fedn.Status(
            status="Client {} connecting to ModelValidationStream.".format(client.name))
        status.log_level = fedn.Status.INFO

        status.sender.name = self.id
        status.sender.role = role_to_proto_role(self.role)

        self._subscribe_client_to_queue(client, fedn.Channel.MODEL_VALIDATIONS)
        q = self.__get_queue(client, fedn.Channel.MODEL_VALIDATIONS)

        self._send_status(status)

        while context.is_active():
            try:
                yield q.get(timeout=1.0)
            except queue.Empty:
                pass

    def ModelValidationRequestStream(self, response, context):
        """ A server stream RPC endpoint (Validation). Messages from client stream.

        :param response: the response
        :type response: :class:`fedn.common.net.grpc.fedn_pb2.ModelValidationRequest`
        :param context: the context
        :type context: :class:`grpc._server._Context`
        """

        client = response.sender
        status = fedn.Status(
            status="Client {} connecting to ModelValidationRequestStream.".format(client.name))
        status.log_level = fedn.Status.INFO
        status.sender.name = self.id
        status.sender.role = role_to_proto_role(self.role)
        status.timestamp = str(datetime.now())

        self._subscribe_client_to_queue(
            client, fedn.Channel.MODEL_VALIDATION_REQUESTS)
        q = self.__get_queue(client, fedn.Channel.MODEL_VALIDATION_REQUESTS)

        self._send_status(status)

        while context.is_active():
            try:
                yield q.get(timeout=1.0)
            except queue.Empty:
                pass

    def SendModelUpdateRequest(self, request, context):
        """ Send a model update request.

        :param request: the request
        :type request: :class:`fedn.common.net.grpc.fedn_pb2.ModelUpdateRequest`
        :param context: the context
        :type context: :class:`grpc._server._Context`
        :return: the response
        :rtype: :class:`fedn.common.net.grpc.fedn_pb2.Response`
        """
        self._send_request(request, fedn.Channel.MODEL_UPDATE_REQUESTS)

        response = fedn.Response()
        response.response = "RECEIVED ModelUpdateRequest from client {}".format(
            request.sender.name)
        return response  # TODO Fill later

    def SendModelUpdate(self, request, context):
        """ Send a model update response.

        :param request: the request
        :type request: :class:`fedn.common.net.grpc.fedn_pb2.ModelUpdate`
        :param context: the context
        :type context: :class:`grpc._server._Context`
        :return: the response
        :rtype: :class:`fedn.common.net.grpc.fedn_pb2.Response`
        """
        self.control.aggregator.on_model_update(request)

        response = fedn.Response()
        response.response = "RECEIVED ModelUpdate {} from client  {}".format(
            response, response.sender.name)
        return response  # TODO Fill later

    def SendModelValidationRequest(self, request, context):
        """ Send a model validation request.

        :param request: the request
        :type request: :class:`fedn.common.net.grpc.fedn_pb2.ModelValidationRequest`
        :param context: the context
        :type context: :class:`grpc._server._Context`
        :return: the response
        :rtype: :class:`fedn.common.net.grpc.fedn_pb2.Response`
        """
        self._send_request(request, fedn.Channel.MODEL_VALIDATION_REQUESTS)

        response = fedn.Response()
        response.response = "RECEIVED ModelValidationRequest from client {}".format(
            request.sender.name)
        return response  # TODO Fill later

    def register_model_validation(self, validation):
        """Register a model validation.

        :param validation: the model validation
        :type validation: :class:`fedn.common.net.grpc.fedn_pb2.ModelValidation`
        """

        self.tracer.report_validation(validation)

    def SendModelValidation(self, request, context):
        """ Send a model validation response.

        :param request: the request
        :type request: :class:`fedn.common.net.grpc.fedn_pb2.ModelValidation`
        :param context: the context
        :type context: :class:`grpc._server._Context`
        :return: the response
        :rtype: :class:`fedn.common.net.grpc.fedn_pb2.Response`
        """
        self.report_status("Recieved ModelValidation from {}".format(request.sender.name),
                           log_level=fedn.Status.INFO)

        self.register_model_validation(request)

        response = fedn.Response()
        response.response = "RECEIVED ModelValidation {} from client  {}".format(
            response, response.sender.name)
        return response

    ####################################################################################################################

    def run(self):
        """ Start the server."""

        print("COMBINER: {} started, ready for requests. ".format(
            self.id), flush=True)
        try:
            while True:
                signal.pause()
        except (KeyboardInterrupt, SystemExit):
            pass
        self.server.stop()
