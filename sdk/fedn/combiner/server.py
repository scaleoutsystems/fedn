import queue
import threading
import uuid
from concurrent import futures
from datetime import datetime, timedelta

import fedn.proto.alliance_pb2 as alliance
import fedn.proto.alliance_pb2_grpc as rpc
import grpc
from fedn.combiner.role import Role
from scaleout.repository.helpers import get_repository

CHUNK_SIZE = 1024 * 1024


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
        self.models = rpc.ModelServiceStub(channel)
        print("ORCHESTRATOR Client: {} connected to {}:{}".format(self.id, address, port))
        threading.Thread(target=self.__listen_to_model_update_stream, daemon=True).start()
        threading.Thread(target=self.__listen_to_model_validation_stream, daemon=True).start()

    def get_model(self, id):
        # self.lock.acquire()
        from io import BytesIO
        data = BytesIO()
        #print("REACHED DOWNLOAD Trying now with id {}".format(id), flush=True)

        #print("TRYING DOWNLOAD 1.", flush=True)
        parts = self.models.Download(alliance.ModelRequest(id=id))
        for part in parts:
            #print("TRYING DOWNLOAD 2.", flush=True)
            if part.status == alliance.ModelStatus.IN_PROGRESS:
                #print("WRITING PART FOR MODEL:{}".format(id), flush=True)
                data.write(part.data)

            if part.status == alliance.ModelStatus.OK:
                #print("DONE WRITING MODEL RETURNING {}".format(id), flush=True)
                # self.lock.release()
                return data
            if part.status == alliance.ModelStatus.FAILED:
                #print("FAILED TO DOWNLOAD MODEL::: bailing!", flush=True)
                return None

    def set_model(self, model, id):
        from io import BytesIO

        if not isinstance(model, BytesIO):
            bt = BytesIO()

            written_total = 0
            for d in model.stream(32 * 1024):
                written = bt.write(d)
                written_total += written

            #print("bytes written {}".format(written_total), flush=True)
        else:
            bt = model

        import sys
        #print("UPLOADING MODEL OF SIZE {}".format(sys.getsizeof(bt)), flush=True)
        bt.seek(0, 0)

        def upload_request_generator(mdl):
            while True:
                b = mdl.read(CHUNK_SIZE)
                #print("Sending chunks!", flush=True)
                if b:
                    result = alliance.ModelRequest(data=b, id=id, status=alliance.ModelStatus.IN_PROGRESS)
                else:
                    result = alliance.ModelRequest(id=id, data=None, status=alliance.ModelStatus.OK)
                yield result
                if not b:
                    break

        result = self.models.Upload(upload_request_generator(bt))

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

class FednServer(rpc.CombinerServicer, rpc.ReducerServicer, rpc.ConnectorServicer, rpc.ModelServiceServicer):
    """ Communication relayer. """

    def __init__(self, connect_config, get_orchestrator):
        self.clients = {}

        import io
        from collections import defaultdict
        self.models = defaultdict(io.BytesIO)
        self.models_metadata = {}
        #self.lock = threading.Lock()

        # self.project = project
        self.role = Role.COMBINER

        from fedn.discovery.connect import DiscoveryCombinerConnect, State
        self.controller = DiscoveryCombinerConnect(connect_config['discover_host'],
                                                   connect_config['discover_port'],
                                                   connect_config['token'],
                                                   connect_config['myhost'],
                                                   connect_config['myport'],
                                                   connect_config['myname'])

        import time
        tries = 90
        status = None
        while True:
            if tries > 0:
                status = self.controller.connect()
                if status == State.Disconnected:
                    tries -= 1

                if status == State.Connected:
                    break

            time.sleep(2)
            print("waiting to reconnect..")

        self.id = connect_config['myname']
        address = connect_config['myhost']
        port = connect_config['myport']

        config, _ = self.controller.get_config()

        self.repository = get_repository(config=config)
        self.bucket_name = config["storage_bucket"]

        # get the appropriate combiner class and instantiate with a pointer to the alliance server instance and repository
        # self.net = OrchestratorClient(address, port, self.id)
        # threading.Thread(target=self.__listen_to_model_update_stream, daemon=True).start()
        # threading.Thread(target=self.__listen_to_model_validation_stream, daemon=True).start()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))

        # TODO setup sevices according to execution context! - That will be really sexy.
        rpc.add_CombinerServicer_to_server(self, self.server)
        rpc.add_ConnectorServicer_to_server(self, self.server)
        rpc.add_ReducerServicer_to_server(self, self.server)
        rpc.add_ModelServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port('[::]:' + str(port))

        self.orchestrator = get_orchestrator(config)(address, port, self.id, self.role, self.repository)

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

    ## Model Service
    def Upload(self, request_iterator, context):
        #print("STARTING UPLOAD!", flush=True)
        result = None
        for request in request_iterator:
            if request.status == alliance.ModelStatus.IN_PROGRESS:
                #print("STARTING UPLOAD: WRITING BYTES", flush=True)
                self.models[request.id].write(request.data)
                self.models_metadata.update({request.id: alliance.ModelStatus.IN_PROGRESS})
                # result = alliance.ModelResponse(id=request.id, status=alliance.ModelStatus.IN_PROGRESS,
                #                                message="Got data successfully.")

            if request.status == alliance.ModelStatus.OK and not request.data:
                #print("TRANSFER OF MODEL IS COMPLETED!!! ", flush=True)
                import sys
                #print(" saved model is size: {}".format(sys.getsizeof(self.models[request.id])))
                result = alliance.ModelResponse(id=request.id, status=alliance.ModelStatus.OK,
                                                message="Got model successfully.")
                self.models_metadata.update({request.id: alliance.ModelStatus.OK})
                return result

    def Download(self, request, context):

        #print("STARTING DOWNLOAD!", flush=True)
        try:
            if self.models_metadata[request.id] != alliance.ModelStatus.OK:
                print("Error file is not ready", flush=True)
                yield alliance.ModelResponse(id=request.id, data=None, status=alliance.ModelStatus.FAILED)
        except Exception as e:
            print("Error file does not exist", flush=True)
            yield alliance.ModelResponse(id=request.id, data=None, status=alliance.ModelStatus.FAILED)

        try:
            from io import BytesIO
            #print("getting object to download on client {}".format(request.id), flush=True)
            obj = self.models[request.id]
            obj.seek(0,0)
            # Have to copy object to not mix up the file pointers when sending... fix in better way.
            obj = BytesIO(obj.read())
            import sys
            #print("got object of size {}".format(sys.getsizeof(obj)), flush=True)
            with obj as f:
                while True:
                    piece = f.read(CHUNK_SIZE)
                    if len(piece) == 0:
                        #print("MODEL {} : Sending last message! ".format(request.id), flush=True)
                        yield alliance.ModelResponse(id=request.id, data=None, status=alliance.ModelStatus.OK)
                        return
                    yield alliance.ModelResponse(id=request.id, data=piece, status=alliance.ModelStatus.IN_PROGRESS)
        except Exception as e:
            print("Downloading went wrong! {}".format(e), flush=True)

    ####################################################################################################################

    def run(self, config):
        print("COMBINER:starting combiner", flush=True)
        # TODO change hostname to configurable and environmental overridable value

        # discovery = DiscoveryCombinerConnect(host=config['discover_host'], port=config['discover_port'],
        #                                     token=config['token'], myhost=self.id, myport=12080, myname=self.id)

        # TODO override by input parameters
        # config = {'round_timeout': timeout, 'seedmodel': seedmodel, 'rounds': rounds, 'active_clients': active,
        #         'discover_host': discoverhost, 'discover_port': discoverport, 'token': token}
        import time
        old_status = "NYD"

        while True:

            status, _ = self.controller.check_status()

            print("COMBINER IN STATE: {} previous {}".format(status, old_status))

            if status == "D":
                print("COMBINER IS DECOMMISONED, report back results and quit")
                return

            if status == "R" and old_status == "S":
                status = self.controller.update_status("I")

            if status == "I" and old_status != "I":
                # if self.orchestrator.satified():
                status = self.controller.update_status("C")
                cfg, _ = self.controller.get_config()
                self.orchestrator.run(cfg)
                ## TODO advertice results?
                ## TODO report executed config
                status = self.controller.update_status("R")

            old_status = status
            # prevent spin
            time.sleep(5)
