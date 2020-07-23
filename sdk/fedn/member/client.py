import json
import os
import tempfile
import threading
from datetime import datetime

import fedn.proto.alliance_pb2 as alliance
import fedn.proto.alliance_pb2_grpc as rpc
import grpc
# TODO Remove from this level. Abstract to unified non implementation specific client.
from fedn.utils.dispatcher import Dispatcher
from scaleout.repository.helpers import get_repository


class Client:

    def __init__(self, config):

        from fedn.discovery.connect import DiscoveryClientConnect, State
        self.controller = DiscoveryClientConnect(config['discover_host'],
                                                 config['discover_port'],
                                                 config['token'],
                                                 config['name'])
        self.name = config['name']

        import time
        tries = 3
        status = None
        while True:
            if tries > 0:
                status = self.controller.connect()
                if status == State.Disconnected:
                    tries -= 1

                if status == State.Connected:
                    break

            time.sleep(5)
            print("waiting to reconnect..")

        combiner = None
        tries = 180
        while True:
            status, state = self.controller.check_status()
            print("got status {}".format(status),flush=True)
            if state == state.Disconnected:
                print("lost connection. trying...")
                tries -= 1
                time.sleep(1)
                if tries < 1:
                    print("ERROR! NO CONTACT!", flush=True)
                    raise Exception("NO CONTACT WITH DISCOVERY NODE")
                self.controller.connect()
            if status != 'A':
                print("waiting to be assigned..", flush=True)
                time.sleep(5)
            if status == 'A':
                print("yay! got assigned, fetching combiner", flush=True)
                combiner, _ = self.controller.get_config()
                break

        # TODO REMOVE ONLY FOR TESTING (only used for partial restructuring)
        repo_config = {'storage_access_key': 'minio',
                       'storage_secret_key': 'minio123',
                       'storage_bucket': 'models',
                       'storage_secure_mode': False,
                       'storage_hostname': 'minio',
                       'storage_port': 9000}

        self.repository = get_repository(repo_config)
        self.bucket_name = repo_config['storage_bucket']

        channel = grpc.insecure_channel(combiner['host'] + ":" + str(combiner['port']))
        self.connection = rpc.ConnectorStub(channel)
        self.orchestrator = rpc.CombinerStub(channel)

        print("Client: {} connected to {}:{}".format(self.name, combiner['host'], combiner['port']))

        # TODO REMOVE OVERRIDE WITH CONTEXT FETCHED
        dispatch_config = {'entry_points':
                               {'predict': {'command': 'python3 predict.py'},
                                'train': {'command': 'python3 train.py'},
                                'validate': {'command': 'python3 validate.py'}}}
        import os

        # TODO REMOVE OVERRIDE WITH CONTEXT FETCHED
        dispatch_dir = os.getcwd()
        self.dispatcher = Dispatcher(dispatch_config, dispatch_dir)
        self.lock = threading.Lock()

        threading.Thread(target=self._send_heartbeat, daemon=True).start()
        threading.Thread(target=self.__listen_to_model_update_request_stream, daemon=True).start()
        threading.Thread(target=self.__listen_to_model_validation_request_stream, daemon=True).start()

    def __listen_to_model_update_request_stream(self):
        """ Subscribe to the model update request stream. """
        r = alliance.ClientAvailableMessage()
        r.sender.name = self.name
        r.sender.role = alliance.WORKER
        metadata = [('client', r.sender.name)]
        for request in self.orchestrator.ModelUpdateRequestStream(r, metadata=metadata):
            if request.sender.role == alliance.COMBINER:
                # Process training request
                global_model_id = request.model_id
                # TODO: Error handling
                self.send_status("Received model update request.", log_level=alliance.Status.AUDIT,
                                 type=alliance.StatusType.MODEL_UPDATE_REQUEST, request=request)
                model_id = self.__process_training_request(global_model_id)

                if model_id != None:
                    # Notify the requesting client that a model update is available
                    update = alliance.ModelUpdate()
                    update.sender.name = self.name
                    update.sender.role = alliance.WORKER
                    update.receiver.name = request.sender.name
                    update.receiver.role = request.sender.role
                    update.model_id = request.model_id
                    update.model_update_id = model_id
                    update.timestamp = str(datetime.now())
                    update.correlation_id = request.correlation_id
                    response = self.orchestrator.SendModelUpdate(update)

                    self.send_status("Model update completed.", log_level=alliance.Status.AUDIT,
                                     type=alliance.StatusType.MODEL_UPDATE, request=update)

                else:
                    self.send_status("Client {} failed to complete model update.", log_level=alliance.Status.WARNING,
                                     request=request)

    def __listen_to_model_validation_request_stream(self):
        """ Subscribe to the model update request stream. """
        r = alliance.ClientAvailableMessage()
        r.sender.name = self.name
        r.sender.role = alliance.WORKER
        for request in self.orchestrator.ModelValidationRequestStream(r):
            # Process training request
            model_id = request.model_id
            # TODO: Error handling
            self.send_status("Recieved model validation request.", log_level=alliance.Status.AUDIT,
                             type=alliance.StatusType.MODEL_VALIDATION_REQUEST, request=request)
            metrics = self.__process_validation_request(model_id)

            if metrics != None:
                # Send validation
                validation = alliance.ModelValidation()
                validation.sender.name = self.name
                validation.sender.role = alliance.WORKER
                validation.receiver.name = request.sender.name
                validation.receiver.role = request.sender.role
                validation.model_id = model_id
                validation.data = json.dumps(metrics)
                self.str = str(datetime.now())
                validation.timestamp = self.str
                validation.correlation_id = request.correlation_id
                response = self.orchestrator.SendModelValidation(validation)
                self.send_status("Model validation completed.", log_level=alliance.Status.AUDIT,
                                 type=alliance.StatusType.MODEL_VALIDATION, request=validation)
            else:
                self.send_status("Client {} failed to complete model validation.".format(self.client),
                                 log_level=alliance.Status.WARNING, request=request)

    def __process_training_request(self, model_id):
        self.send_status("\t Processing training request for model_id {}".format(model_id))
        try:
            model = self.repository.get_model(model_id)
            fid, infile_name = tempfile.mkstemp(suffix='.h5')
            fod, outfile_name = tempfile.mkstemp(suffix='.h5')

            with open(infile_name, "wb") as fh:
                fh.write(model)

            self.dispatcher.run_cmd("train {} {}".format(infile_name, outfile_name))

            model_id = self.repository.set_model(outfile_name, is_file=True)

            os.unlink(infile_name)
            os.unlink(outfile_name)

        except Exception as e:
            print("ERROR could not process training request due to error: {}".format(e))
            model_id = None

        return model_id

    def __process_validation_request(self, model_id):
        self.send_status("Processing validation request for model_id {}".format(model_id))

        try:
            model = self.repository.get_model(model_id)
            fid, infile_name = tempfile.mkstemp(suffix='.h5')
            fod, outfile_name = tempfile.mkstemp(suffix='.h5')
            with open(infile_name, "wb") as fh:
                fh.write(model)

            self.dispatcher.run_cmd("validate {} {}".format(infile_name, outfile_name))

            with open(outfile_name, "r") as fh:
                validation = json.loads(fh.read())

            os.unlink(infile_name)
            os.unlink(outfile_name)

            return validation
        except Exception as e:
            print("Validation failed with exception {}".format(e), flush=True)
            return None

    def send_status(self, msg, log_level=alliance.Status.INFO, type=None, request=None):
        from google.protobuf.json_format import MessageToJson
        status = alliance.Status()

        status.sender.name = self.name
        status.sender.role = alliance.WORKER
        status.log_level = log_level
        status.status = str(msg)
        if type is not None:
            status.type = type

        if request is not None:
            status.data = MessageToJson(request)

        response = self.connection.SendStatus(status)

    def _send_heartbeat(self, update_frequency=2.0):
        while True:
            heartbeat = alliance.Heartbeat(sender=alliance.Client(name=self.name, role=alliance.WORKER))
            self.connection.SendHeartbeat(heartbeat)
            # self.send_status("HEARTBEAT from {}".format(self.client),log_level=alliance.Status.INFO)
            import time
            time.sleep(update_frequency)

    def run(self):
        import time
        try:
            while True:
                time.sleep(1)
                print("CLIENT running.", flush=True)
        except KeyboardInterrupt:
            print("ok exiting..")
