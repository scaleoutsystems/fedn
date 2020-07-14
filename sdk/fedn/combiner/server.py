from concurrent import futures

import grpc
import time
import uuid
import queue
import threading

import fedn.proto.alliance_pb2 as alliance
import fedn.proto.alliance_pb2_grpc as rpc

from datetime import datetime, timedelta
from scaleout.repository.helpers import get_repository
from fedn.utils.mongo import connect_to_mongodb

from fedn.combiner.role import Role


####################################################################################################################

# class PredictionServer:
#    #TODO add a flask api and run in separate thread.
#    pass
def whoami(client, instance):
    client.name = instance.id
    client.role = role_to_proto_role(instance.role)
    return client


def role_to_proto_role(role):
    if role == Role.COMBINER:
        return alliance.COMBINER
    if role == Role.WORKER:
        return alliance.WORKER
    if role == Role.REDUCER:
        return alliance.REDUCER
    if role == Role.OTHER:
        return alliance.OTHER


class CombinerClient:
    def __init__(self, address, port, id, role):

        self.id = id
        self.role = role

        channel = grpc.insecure_channel(address + ":" + str(port))
        self.connection = rpc.ConnectorStub(channel)
        self.orchestrator = rpc.CombinerStub(channel)
        print("ORCHESTRATOR Client: {} connected to {}:{}".format(self.id, address, port))
        threading.Thread(target=self.__listen_to_model_update_stream, daemon=True).start()
        threading.Thread(target=self.__listen_to_model_validation_stream, daemon=True).start()

    def __listen_to_model_update_stream(self):
        """ Subscribe to the model update request stream. """
        r = alliance.ClientAvailableMessage()

        whoami(r.sender, self)

        for request in self.orchestrator.ModelUpdateStream(r):
            # A client sent a model update to be handled by the combiner
            if request.client.name != "reducer":
                print("ORCHESTRATOR: received model from client! {}".format(request.client), flush=True)
                self.receive_model_candidate(request.model_update_id)
            print("Recieved model update.", flush=True)

    def __listen_to_model_validation_stream(self):
        """ Subscribe to the model update request stream. """
        r = alliance.ClientAvailableMessage()
        whoami(r.sender, self)
        for validation in self.orchestrator.ModelValidationStream(r):
            # A client sent a model update to be handled by the combiner
            self.receive_validation(validation)
            print("Recieved model validation.", flush=True)

    def request_model_update(self, model_id, clients=[]):
        """ Ask members in from_clients list to update the current global model. """
        print("ORCHESTRATOR: Sending to clients {}".format(clients), flush=True)
        request = alliance.ModelUpdateRequest()
        whoami(request.sender, self)
        request.model_id = model_id
        request.correlation_id = str(uuid.uuid4())
        request.timestamp = str(datetime.now())

        if len(clients) == 0:
            # Broadcast request to all active member clients
            request.receiver.name = ""
            request.receiver.role = alliance.WORKER
            response = self.orchestrator.SendModelUpdateRequest(request)

        else:
            # Send to all specified clients
            for client in clients:
                request.receiver.name = client.name
                request.receiver.role = alliance.WORKER
                self.orchestrator.SendModelUpdateRequest(request)

        print("Requesting model update from clients {}".format(clients), flush=True)

    def request_model_validation(self, model_id, from_clients=[]):
        """ Send a request for members in from_client to validate the model <model_id>.
            The default is to broadcast the request to all active members.
        """
        request = alliance.ModelValidationRequest()
        whoami(request.sender, self)
        request.model_id = model_id
        request.correlation_id = str(uuid.uuid4())
        request.timestamp = str(datetime.now())

        if len(from_clients) == 0:
            request.receiver.name = ""  # Broadcast request to all active member clients
            request.receiver.role = alliance.WORKER
            self.orchestrator.SendModelValidationRequest(request)
        else:
            # Send to specified clients
            for client in from_clients:
                request.receiver.name = client.name
                request.receiver.role = alliance.WORKER
                self.orchestrator.SendModelValidationRequest(request)

        print("ORCHESTRATOR: Sent validation request for model {}".format(model_id), flush=True)

    def _list_clients(self, channel):
        request = alliance.ListClientsRequest()
        whoami(request.sender, self)
        request.channel = channel
        clients = self.connection.ListActiveClients(request)
        return clients.client

    def get_active_trainers(self):
        trainers = self._list_clients(alliance.Channel.MODEL_UPDATE_REQUESTS)
        return trainers

    def get_active_validators(self):
        validators = self._list_clients(alliance.Channel.MODEL_VALIDATION_REQUESTS)
        return validators

    def nr_active_trainers(self):
        return len(self.get_active_trainers())

    def nr_active_validators(self):
        return len(self.get_active_validators())


####################################################################################################################
####################################################################################################################

class FednServer(rpc.CombinerServicer, rpc.ReducerServicer, rpc.ConnectorServicer):
    """ Communication relayer. """

    def __init__(self, project, get_orchestrator):
        self.clients = {}

        self.project = project
        self.role = Role.COMBINER
        self.id = "combiner"

        address = "localhost"
        port = 12808
        try:
            unpack = project.config['Alliance']
            address = unpack['controller_host']
            port = unpack['controller_port']
            # self.client = unpack['Member']['name']
        except KeyError as e:
            print("ORCHESTRATOR: could not get all values from config file {}".format(e))

        try:
            unpack = self.project.config['Alliance']
            address = unpack['controller_host']
            port = unpack['controller_port']

            self.repository = get_repository(config=unpack['Repository'])
            self.bucket_name = unpack["Repository"]["minio_bucket"]

        except KeyError as e:
            print("ORCHESETRATOR: could not get all values from config file {}".format(e), flush=True)

        # get the appropriate combiner class and instantiate with a pointer to the alliance server instance and repository
        # self.net = OrchestratorClient(address, port, self.id)
        # threading.Thread(target=self.__listen_to_model_update_stream, daemon=True).start()
        # threading.Thread(target=self.__listen_to_model_validation_stream, daemon=True).start()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))
        # TODO refactor services into separate services
        rpc.add_CombinerServicer_to_server(self, self.server)
        rpc.add_ConnectorServicer_to_server(self, self.server)
        rpc.add_ReducerServicer_to_server(self, self.server)
        self.server.add_insecure_port('[::]:' + str(port))

        self.orchestrator = get_orchestrator(project)(address, port, self.id, self.role, self.repository)

        self.server.start()

    # def __get_clients(self):
    #    return self.clients

    def __join_client(self, client):
        if not client.name in self.clients.keys():
            self.clients[client.name] = {"lastseen": datetime.now()}
            print("New client connected:{}".format(client), flush=True)

    def _subscribe_client_to_queue(self, client, queue_name):
        self.__join_client(client)
        if not queue_name in self.clients[client.name].keys():
            self.clients[client.name][queue_name] = queue.Queue()

    def __get_queue(self, client, queue_name):
        try:
            return self.clients[client.name][queue_name]
        except KeyError:
            raise

    def __get_status_queue(self, client):
        return self.__get_queue(client, alliance.Channel.STATUS)

    def _send_request(self, request, queue_name):
        self.__route_request_to_client(request, request.receiver, queue_name)

    def _broadcast_request(self, request, queue_name):
        """ Publish a request to all subscribed members. """
        active_clients = self._list_active_clients()
        for client in active_clients:
            self.clients[client.name][queue_name].put(request)

    def __route_request_to_client(self, request, client, queue_name):
        try:
            q = self.__get_queue(client, queue_name)
            q.put(request)
        except:
            print("Failed to route request to client: {} {}", request.receiver, queue_name)
            raise

    def _send_status(self, status):
        for name, client in self.clients.items():
            try:
                q = client[alliance.Channel.STATUS]
                status.timestamp = str(datetime.now())
                q.put(status)
            except KeyError:
                pass

    def __register_heartbeat(self, client):
        """ Adds a client entry in the clients dict if first time connecting.
            Updates heartbeat timestamp.
        """
        self.__join_client(client)
        self.clients[client.name]["lastseen"] = datetime.now()

    def AllianceStatusStream(self, response, context):
        """ A server stream RPC endpoint that emits status messages. """
        status = alliance.Status(status="Client {} connecting to AllianceStatusStream.".format(response.sender))
        status.log_level = alliance.Status.INFO
        status.sender.name = self.id
        status.sender.role = role_to_proto_role(self.role)
        self._subscribe_client_to_queue(response.sender, alliance.Channel.STATUS)
        q = self.__get_queue(response.sender, alliance.Channel.STATUS)
        self._send_status(status)

        while True:
            yield q.get()

    def SendStatus(self, status: alliance.Status, context):
        # Register a heartbeat (if the clients sends a message it is online)
        # self.__register_heartbeat(status.client)
        # Add the status message to all subscribers of the status channel
        self._send_status(status)

        response = alliance.Response()
        response.response = "Status received."
        return response

    def _list_subscribed_clients(self, queue_name):
        subscribed_clients = []
        for name, client in self.clients.items():
            if queue_name in client.keys():
                subscribed_clients.append(name)
        return subscribed_clients

    def _list_active_clients(self, channel):
        active_clients = []
        for client in self._list_subscribed_clients(channel):
            # This can break with different timezones. 
            now = datetime.now()
            then = self.clients[client]["lastseen"]
            # TODO: move the heartbeat timeout to config. 
            if (now - then) < timedelta(seconds=30):
                active_clients.append(client)
        return active_clients

    def ListActiveClients(self, request: alliance.ListClientsRequest, context):
        """ RPC endpoint that returns a ClientList containing the names of all active clients. 
            An active client has sent a status message / responded to a heartbeat
            request in the last 10 seconds.  
        """
        clients = alliance.ClientList()
        active_clients = self._list_active_clients(request.channel)

        for client in active_clients:
            clients.client.append(alliance.Client(name=client, role=alliance.WORKER))
        return clients

    def SendHeartbeat(self, heartbeat: alliance.Heartbeat, context):
        """ RPC that lets clients send a hearbeat, notifying the server that 
            the client is available. """
        self.__register_heartbeat(heartbeat.sender)
        response = alliance.Response()
        response.sender.name = heartbeat.sender.name
        response.sender.role = heartbeat.sender.role
        response.response = "Heartbeat received"
        return response

    ## Combiner Service

    def ModelUpdateStream(self, update, context):
        client = update.sender
        status = alliance.Status(status="Client {} connecting to ModelUpdateStream.".format(client.name))
        status.log_level = alliance.Status.INFO
        status.sender.name = self.id
        status.sender.role = role_to_proto_role(self.role)

        self._subscribe_client_to_queue(client, alliance.Channel.MODEL_UPDATES)
        q = self.__get_queue(client, alliance.Channel.MODEL_UPDATES)

        self._send_status(status)

        while True:
            yield q.get()

    def ModelUpdateRequestStream(self, response, context):
        """ A server stream RPC endpoint. Messages from client stream. """

        client = response.sender
        metadata = context.invocation_metadata()
        if metadata:
            print("\n\n\nGOT METADATA: {}\n\n\n".format(metadata), flush=True)

        status = alliance.Status(status="Client {} connecting to ModelUpdateRequestStream.".format(client.name))
        status.log_level = alliance.Status.INFO

        whoami(status.sender, self)
        # print("Client {} connecting to ModelUpdateRequestStream.".format(client))

        self._subscribe_client_to_queue(client, alliance.Channel.MODEL_UPDATE_REQUESTS)
        q = self.__get_queue(client, alliance.Channel.MODEL_UPDATE_REQUESTS)

        self._send_status(status)

        while True:
            yield q.get()

    def ModelValidationStream(self, update, context):
        client = update.sender
        status = alliance.Status(status="Client {} connecting to ModelValidationStream.".format(client.name))
        status.log_level = alliance.Status.INFO

        status.sender.name = self.id
        status.sender.role = role_to_proto_role(self.role)

        # print("Client {} connecting to ModelUpdateStream.".format(client))
        self._subscribe_client_to_queue(client, alliance.Channel.MODEL_VALIDATIONS)
        q = self.__get_queue(client, alliance.Channel.MODEL_VALIDATIONS)

        self._send_status(status)

        while True:
            yield q.get()

    def ModelValidationRequestStream(self, response, context):
        """ A server stream RPC endpoint. Messages from client stream. """

        client = response.sender
        status = alliance.Status(status="Client {} connecting to ModelValidationRequestStream.".format(client.name))
        status.log_level = alliance.Status.INFO
        status.sender.name = self.id
        status.sender.role = role_to_proto_role(self.role)
        # whoami(status.sender, self)

        self._subscribe_client_to_queue(client, alliance.Channel.MODEL_VALIDATION_REQUESTS)
        q = self.__get_queue(client, alliance.Channel.MODEL_VALIDATION_REQUESTS)

        self._send_status(status)

        while True:
            yield q.get()

    def SendModelUpdateRequest(self, request, context):
        """ Send a model update request. """
        self._send_request(request, alliance.Channel.MODEL_UPDATE_REQUESTS)

        response = alliance.Response()
        response.response = "CONTROLLER RECEIVED ModelUpdateRequest from client {}".format(request.sender.name)
        return response  # TODO Fill later

    def SendModelUpdate(self, request, context):
        """ Send a model update response. """
        # self._send_request(request,alliance.Channel.MODEL_UPDATES)
        self.orchestrator.receive_model_candidate(request.model_update_id)
        print("ORCHESTRATOR: Received model update", flush=True)

        response = alliance.Response()
        response.response = "RECEIVED ModelUpdate {} from client  {}".format(response, response.sender.name)
        return response  # TODO Fill later

    def SendModelValidationRequest(self, request, context):
        """ Send a model update request. """
        self._send_request(request, alliance.Channel.MODEL_VALIDATION_REQUESTS)

        response = alliance.Response()
        response.response = "CONTROLLER RECEIVED ModelValidationRequest from client {}".format(request.sender.name)
        return response  # TODO Fill later

    def SendModelValidation(self, request, context):
        """ Send a model update response. """
        # self._send_request(request,alliance.Channel.MODEL_VALIDATIONS)
        self.orchestrator.receive_validation(request)
        print("ORCHESTRATOR received validation ", flush=True)
        response = alliance.Response()
        response.response = "RECEIVED ModelValidation {} from client  {}".format(response, response.sender.name)
        return response  # TODO Fill later

    ## Reducer Service

    def GetGlobalModel(self, request, context):

        print("got globalmodel request, sending response! ", flush=True)
        response = alliance.GetGlobalModelResponse()
        whoami(response.sender, self)
        response.receiver.name = "reducer"
        response.receiver.role = role_to_proto_role(Role.REDUCER)
        response.model_id = self.orchestrator.get_model_id()
        return response

    ####################################################################################################################

    def run(self, config):
        print("ORCHESTRATOR:starting combiner", flush=True)

        self.orchestrator.run(config)
