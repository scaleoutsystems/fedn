import os
import queue
import threading
import uuid
from datetime import datetime, timedelta
from fedn.common.net.grpc.server import Server
import fedn.common.net.grpc.fedn_pb2 as fedn
import fedn.common.net.grpc.fedn_pb2_grpc as rpc
from fedn.clients.combiner.modelservice import ModelService
from fedn.common.storage.s3.s3repo import S3ModelRepository

# from fedn.combiner.role import Role

from enum import Enum


class Role(Enum):
    WORKER = 1
    COMBINER = 2
    REDUCER = 3
    OTHER = 4


def role_to_proto_role(role):
    if role == Role.COMBINER:
        return fedn.COMBINER
    if role == Role.WORKER:
        return fedn.WORKER
    if role == Role.REDUCER:
        return fedn.REDUCER
    if role == Role.OTHER:
        return fedn.OTHER


####################################################################################################################
####################################################################################################################

class Combiner(rpc.CombinerServicer, rpc.ReducerServicer, rpc.ConnectorServicer, rpc.ControlServicer):
    """ Communication relayer. """

    def __init__(self, connect_config):
        self.clients = {}

        import io
        from collections import defaultdict
        self.modelservice = ModelService()

        self.model_id = None

        self.role = Role.COMBINER

        self.id = connect_config['myname']
        address = connect_config['myhost']
        port = connect_config['myport']

        self.max_clients = connect_config['max_clients']

        
        from fedn.common.net.connect import ConnectorCombiner, Status
        announce_client = ConnectorCombiner(host=connect_config['discover_host'],
                                            port=connect_config['discover_port'],
                                            myhost=connect_config['myhost'],
                                            myport=connect_config['myport'],
                                            token=connect_config['token'],
                                            name=connect_config['myname'])

        import time
        response = None
        while True:
            status, response = announce_client.announce()
            if status == Status.TryAgain:
                time.sleep(5)
                continue
            if status == Status.Assigned:
                config = response
                print("COMBINER: was announced successfully. Waiting for clients and commands!", flush=True)
                break

        import base64
        cert = base64.b64decode(response['certificate'])  # .decode('utf-8')
        key = base64.b64decode(response['key'])  # .decode('utf-8')

        grpc_config = {'port': port,
                       'secure': connect_config['secure'],
                       'certificate': cert,
                       'key': key}

        # TODO remove temporary hardcoded config of storage persistance backend
        combiner_config = {'storage_access_key': os.environ['FEDN_MINIO_ACCESS_KEY'],
                           'storage_secret_key': os.environ['FEDN_MINIO_SECRET_KEY'],
                           'storage_bucket': 'models',
                           'storage_secure_mode': False,
                           'storage_hostname': os.environ['FEDN_MINIO_HOST'],
                           'storage_port': int(os.environ['FEDN_MINIO_PORT'])}

        self.repository = S3ModelRepository(combiner_config)
        self.bucket_name = combiner_config["storage_bucket"]

        self.server = Server(self,self.modelservice, grpc_config)

        # The handler that implements the particular combination strategy used by this combiner instance. 
        # TODO: Make configurable on start.
        from fedn.algo.fedavg import FEDAVGCombiner
        self.combiner = FEDAVGCombiner(self.id, self.repository, self, self.modelservice)
        threading.Thread(target=self.combiner.run, daemon=True).start()

        from fedn.common.tracer.mongotracer import MongoTracer
        self.tracer = MongoTracer()

        self.server.start()

    def __whoami(self, client, instance):

        def role_to_proto_role(role):
            if role == Role.COMBINER:
                return fedn.COMBINER
            if role == Role.WORKER:
                return fedn.WORKER
            if role == Role.REDUCER:
                return fedn.REDUCER
            if role == Role.OTHER:
                return fedn.OTHER

        client.name = instance.id
        client.role = role_to_proto_role(instance.role)
        return client

    def get_active_model(self):
        return self.model_id

    def set_active_model(self, model_id):
        self.model_id = model_id

    def request_model_update(self, model_id, clients=[]):
        """ Ask members in from_clients list to update the current global model. """
        print("COMBINER: Sending to clients {}".format(clients), flush=True)
        request = fedn.ModelUpdateRequest()
        self.__whoami(request.sender, self)
        request.model_id = model_id
        request.correlation_id = str(uuid.uuid4())
        request.timestamp = str(datetime.now())

        if len(clients) == 0:
            # Broadcast request to all active member clients
            request.receiver.name = ""
            request.receiver.role = fedn.WORKER
            response = self.SendModelUpdateRequest(request, self)

        else:
            # Send to all specified clients
            for client in clients:
                request.receiver.name = client.name
                request.receiver.role = fedn.WORKER
                self.SendModelUpdateRequest(request, self)
        # print("Requesting model update from clients {}".format(clients), flush=True)

    def request_model_validation(self, model_id, from_clients=[]):
        """ Send a request for members in from_client to validate the model <model_id>.
            The default is to broadcast the request to all active members.
        """
        request = fedn.ModelValidationRequest()
        self.__whoami(request.sender, self)
        request.model_id = model_id
        request.correlation_id = str(uuid.uuid4())
        request.timestamp = str(datetime.now())

        if len(from_clients) == 0:
            request.receiver.name = ""  # Broadcast request to all active member clients
            request.receiver.role = fedn.WORKER
            self.SendModelValidationRequest(request, self)
        else:
            # Send to specified clients
            for client in from_clients:
                request.receiver.name = client.name
                request.receiver.role = fedn.WORKER
                self.SendModelValidationRequest(request, self)

        print("COMBINER: Sent validation request for model {}".format(model_id), flush=True)

    def _list_clients(self, channel):
        request = fedn.ListClientsRequest()
        self.__whoami(request.sender, self)
        request.channel = channel
        clients = self.ListActiveClients(request, self)
        return clients.client

    def get_active_trainers(self):
        trainers = self._list_clients(fedn.Channel.MODEL_UPDATE_REQUESTS)
        return trainers

    def get_active_validators(self):
        validators = self._list_clients(fedn.Channel.MODEL_VALIDATION_REQUESTS)
        return validators

    def nr_active_trainers(self):
        return len(self.get_active_trainers())

    def nr_active_validators(self):
        return len(self.get_active_validators())

    ####################################################################################################################

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
        return self.__get_queue(client, fedn.Channel.STATUS)

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

        self.tracer.report(status)
        for name, client in self.clients.items():
            try:
                q = client[fedn.Channel.STATUS]
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

    #####################################################################################################################

    ## Control Service

    def Start(self, control: fedn.ControlRequest, context):
        response = fedn.ControlResponse()
        print("\n\n\n GOT CONTROL **START** from Command {}\n\n\n".format(control.command), flush=True)

        config = {}
        for parameter in control.parameter:
            config.update({parameter.key: parameter.value})
        print("\n\n\n\nSTARTING JOB AT COMBINER WITH {}\n\n\n\n".format(config), flush=True)

        job_id = self.combiner.push_run_config(config)
        return response

    def Configure(self, control: fedn.ControlRequest, context):
        response = fedn.ControlResponse()
        for parameter in control.parameter:
            setattr(self, parameter.key, parameter.value)
        return response

    def Stop(self, control: fedn.ControlRequest, context):
        response = fedn.ControlResponse()
        print("\n\n\n\n\n GOT CONTROL **STOP** from Command\n\n\n\n\n", flush=True)
        return response

    def Report(self, control: fedn.ControlRequest, context):
        """ Descibe current state of the Combiner. """
        response = fedn.ControlResponse()
        print("\n\n\n\n\n GOT CONTROL **REPORT** from Command\n\n\n\n\n", flush=True)

        active_clients = self._list_active_clients(fedn.Channel.MODEL_UPDATE_REQUESTS)
        nr_active_clients = len(active_clients)

        p = response.parameter.add()
        p.key = "nr_active_clients"
        p.value = str(nr_active_clients)
        
        p = response.parameter.add()
        p.key = "model_id"
        model_id = self.get_active_model()
        if model_id == None:
            model_id = ""
        p.value = str(model_id)

        p = response.parameter.add()
        p.key = "nr_unprocessed_tasks"
        p.value = str(len(self.combiner.run_configs))
        
        return response

    #####################################################################################################################

    def AllianceStatusStream(self, response, context):
        """ A server stream RPC endpoint that emits status messages. """
        status = fedn.Status(status="Client {} connecting to AllianceStatusStream.".format(response.sender))
        status.log_level = fedn.Status.INFO
        status.sender.name = self.id
        status.sender.role = role_to_proto_role(self.role)
        self._subscribe_client_to_queue(response.sender, fedn.Channel.STATUS)
        q = self.__get_queue(response.sender, fedn.Channel.STATUS)
        self._send_status(status)

        while True:
            yield q.get()

    def SendStatus(self, status: fedn.Status, context):
        # Add the status message to all subscribers of the status channel
        self._send_status(status)

        response = fedn.Response()
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
            if (now - then) < timedelta(seconds=10):
                active_clients.append(client)
        return active_clients

    def ListActiveClients(self, request: fedn.ListClientsRequest, context):
        """ RPC endpoint that returns a ClientList containing the names of all active clients.
            An active client has sent a status message / responded to a heartbeat
            request in the last 10 seconds.
        """
        clients = fedn.ClientList()
        active_clients = self._list_active_clients(request.channel)

        for client in active_clients:
            clients.client.append(fedn.Client(name=client, role=fedn.WORKER))
        return clients


    def AcceptingClients(self, request: fedn.ConnectionRequest, context):
        response = fedn.ConnectionResponse()
        active_clients = self._list_active_clients(fedn.Channel.MODEL_UPDATE_REQUESTS)

        try:
            #requested = int(self.combiner.config['clients_requested'])
            requested = int(self.max_clients)
            if len(active_clients) >= requested:
                response.status = fedn.ConnectionStatus.NOT_ACCEPTING
                return response
            if len(active_clients) < requested:
                response.status = fedn.ConnectionStatus.ACCEPTING
                return response

        except Exception as e:
            print("Combiner not properly configured!", flush=True)
            raise

        response.status = fedn.ConnectionStatus.TRY_AGAIN_LATER
        return response

    def SendHeartbeat(self, heartbeat: fedn.Heartbeat, context):
        """ RPC that lets clients send a hearbeat, notifying the server that
            the client is available. """
        self.__register_heartbeat(heartbeat.sender)
        response = fedn.Response()
        response.sender.name = heartbeat.sender.name
        response.sender.role = heartbeat.sender.role
        response.response = "Heartbeat received"
        return response

    ## Combiner Service

    def ModelUpdateStream(self, update, context):
        client = update.sender
        status = fedn.Status(status="Client {} connecting to ModelUpdateStream.".format(client.name))
        status.log_level = fedn.Status.INFO
        status.sender.name = self.id
        status.sender.role = role_to_proto_role(self.role)

        self._subscribe_client_to_queue(client, fedn.Channel.MODEL_UPDATES)
        q = self.__get_queue(client, fedn.Channel.MODEL_UPDATES)

        self._send_status(status)

        while True:
            yield q.get()

    def ModelUpdateRequestStream(self, response, context):
        """ A server stream RPC endpoint. Messages from client stream. """

        client = response.sender
        metadata = context.invocation_metadata()
        if metadata:
            print("\n\n\nGOT METADATA: {}\n\n\n".format(metadata), flush=True)

        status = fedn.Status(status="Client {} connecting to ModelUpdateRequestStream.".format(client.name))
        status.log_level = fedn.Status.INFO

        self.__whoami(status.sender, self)

        self._subscribe_client_to_queue(client, fedn.Channel.MODEL_UPDATE_REQUESTS)
        q = self.__get_queue(client, fedn.Channel.MODEL_UPDATE_REQUESTS)

        self._send_status(status)

        while True:
            yield q.get()

    def ModelValidationStream(self, update, context):
        client = update.sender
        status = fedn.Status(status="Client {} connecting to ModelValidationStream.".format(client.name))
        status.log_level = fedn.Status.INFO

        status.sender.name = self.id
        status.sender.role = role_to_proto_role(self.role)

        self._subscribe_client_to_queue(client, fedn.Channel.MODEL_VALIDATIONS)
        q = self.__get_queue(client, fedn.Channel.MODEL_VALIDATIONS)

        self._send_status(status)

        while True:
            yield q.get()

    def ModelValidationRequestStream(self, response, context):
        """ A server stream RPC endpoint. Messages from client stream. """

        client = response.sender
        status = fedn.Status(status="Client {} connecting to ModelValidationRequestStream.".format(client.name))
        status.log_level = fedn.Status.INFO
        status.sender.name = self.id
        status.sender.role = role_to_proto_role(self.role)

        self._subscribe_client_to_queue(client, fedn.Channel.MODEL_VALIDATION_REQUESTS)
        q = self.__get_queue(client, fedn.Channel.MODEL_VALIDATION_REQUESTS)

        self._send_status(status)

        while True:
            yield q.get()

    def SendModelUpdateRequest(self, request, context):
        """ Send a model update request. """
        self._send_request(request, fedn.Channel.MODEL_UPDATE_REQUESTS)

        response = fedn.Response()
        response.response = "CONTROLLER RECEIVED ModelUpdateRequest from client {}".format(request.sender.name)
        return response  # TODO Fill later

    def SendModelUpdate(self, request, context):
        """ Send a model update response. """
        # self._send_request(request,fedn.Channel.MODEL_UPDATES)
        self.combiner.receive_model_candidate(request.model_update_id)
        print("ORCHESTRATOR: Received model update", flush=True)

        response = fedn.Response()
        response.response = "RECEIVED ModelUpdate {} from client  {}".format(response, response.sender.name)
        return response  # TODO Fill later

    def SendModelValidationRequest(self, request, context):
        """ Send a model update request. """
        self._send_request(request, fedn.Channel.MODEL_VALIDATION_REQUESTS)

        response = fedn.Response()
        response.response = "CONTROLLER RECEIVED ModelValidationRequest from client {}".format(request.sender.name)
        return response  # TODO Fill later

    def SendModelValidation(self, request, context):
        """ Send a model update response. """
        # self._send_request(request,fedn.Channel.MODEL_VALIDATIONS)
        self.combiner.receive_validation(request)
        print("ORCHESTRATOR received validation ", flush=True)
        response = fedn.Response()
        response.response = "RECEIVED ModelValidation {} from client  {}".format(response, response.sender.name)
        return response  # TODO Fill later

    ## Reducer Service

    def GetGlobalModel(self, request, context):

        response = fedn.GetGlobalModelResponse()
        self.__whoami(response.sender, self)
        response.receiver.name = "reducer"
        response.receiver.role = role_to_proto_role(Role.REDUCER)
        if not self.get_active_model():
            response.model_id = ''
        else:
            response.model_id = self.get_active_model()
        return response



    ####################################################################################################################

    def run(self):
        import signal
        print("COMBINER:starting combiner", flush=True)
        try:
            while True:
                signal.pause()
        except (KeyboardInterrupt, SystemExit):
            pass
        self.server.stop()
