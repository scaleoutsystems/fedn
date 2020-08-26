import json
import os
import tempfile
import threading

import fedn.proto.alliance_pb2 as alliance
import fedn.proto.alliance_pb2_grpc as rpc
import grpc
# TODO Remove from this level. Abstract to unified non implementation specific client.
from fedn.utils.dispatcher import Dispatcher
from scaleout.repository.helpers import get_repository

CHUNK_SIZE = 1024 * 1024

from enum import Enum
from datetime import datetime


class ClientState(Enum):
    idle = 1
    training = 2
    validating = 3


def ClientStateToString(state):
    if state == ClientState.idle:
        return "IDLE"
    if state == ClientState.training:
        return "TRAINING"
    if state == ClientState.validating:
        return "VALIDATING"

    return "UNKNOWN"


class Client:

    def __init__(self, config):

        from fedn.discovery.connect import DiscoveryClientConnect, State
        self.controller = DiscoveryClientConnect(config['discover_host'],
                                                 config['discover_port'],
                                                 config['token'],
                                                 config['name'],
                                                 config['id'])
        self.name = config['name']

        self.started_at = datetime.now()
        self.logs = []

        import time
        tries = 90
        status = None
        while True:
            if tries > 0:
                status = self.controller.connect()
                if status == State.Disconnected:
                    tries = tries - 1

                if status == State.Connected:
                    break

            time.sleep(2)
            print("try to reconnect to CONTROLLER", flush=True)

        combiner = None
        tries = 180
        while True:
            status, state = self.controller.check_status()
            print("got status {}".format(status), flush=True)
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
        self.models = rpc.ModelServiceStub(channel)

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

        self.state = ClientState.idle

    def get_model(self, id):

        from io import BytesIO
        data = BytesIO()
        # print("REACHED DOWNLOAD Trying now with id {}".format(id), flush=True)

        # print("TRYING DOWNLOAD 1.", flush=True)
        for part in self.models.Download(alliance.ModelRequest(id=id)):

            # print("TRYING DOWNLOAD 2.", flush=True)
            if part.status == alliance.ModelStatus.IN_PROGRESS:
                # print("WRITING PART FOR MODEL:{}".format(id), flush=True)
                data.write(part.data)

            if part.status == alliance.ModelStatus.OK:
                # print("DONE WRITING MODEL RETURNING {}".format(id), flush=True)

                return data
            if part.status == alliance.ModelStatus.FAILED:
                # print("FAILED TO DOWNLOAD MODEL::: bailing!",flush=True)
                return None
        # print("ERROR NO PARTS!",flush=True)
        return data

    def set_model(self, model, id):

        from io import BytesIO

        if not isinstance(model, BytesIO):
            bt = BytesIO()

            for d in model.stream(32 * 1024):
                bt.write(d)
        else:
            bt = model

        # print("SETTING MODEL OF SIZE {}".format(sys.getsizeof(bt)), flush=True)
        bt.seek(0, 0)

        def upload_request_generator(mdl):
            i = 1
            while True:
                b = mdl.read(CHUNK_SIZE)
                if b:
                    result = alliance.ModelRequest(data=b, id=id, status=alliance.ModelStatus.IN_PROGRESS)
                else:
                    result = alliance.ModelRequest(id=id, status=alliance.ModelStatus.OK)

                yield result
                if not b:
                    break

        result = self.models.Upload(upload_request_generator(bt))

        return result

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
                    update.model_update_id = str(model_id)
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
                validation.model_id = str(model_id)
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
        self.state = ClientState.training
        try:
            # print("IN TRAINING REQUEST 1", flush=True)
            mdl = self.get_model(str(model_id))
            import sys
            # print("did i get a model? model_id: {} size:{}".format(model_id, sys.getsizeof(mdl)))
            # print("IN TRAINING REQUEST 2", flush=True)
            # model = self.repository.get_model(model_id)
            fid, infile_name = tempfile.mkstemp(suffix='.h5')
            fod, outfile_name = tempfile.mkstemp(suffix='.h5')

            with open(infile_name, "wb") as fh:
                fh.write(mdl.getbuffer())
            # print("IN TRAINING REQUEST 3", flush=True)
            self.dispatcher.run_cmd("train {} {}".format(infile_name, outfile_name))
            # print("IN TRAINING REQUEST 4", flush=True)
            # model_id = self.repository.set_model(outfile_name, is_file=True)

            import io
            out_model = None
            with open(outfile_name, "rb") as fr:
                out_model = io.BytesIO(fr.read())
            # print("IN TRAINING REQUEST 5", flush=True)
            import uuid
            model_id = uuid.uuid4()
            self.set_model(out_model, str(model_id))
            # print("IN TRAINING REQUEST 6", flush=True)
            os.unlink(infile_name)
            os.unlink(outfile_name)

        except Exception as e:
            print("ERROR could not process training request due to error: {}".format(e))
            model_id = None

        self.state = ClientState.idle

        return model_id

    def __process_validation_request(self, model_id):
        self.send_status("Processing validation request for model_id {}".format(model_id))
        self.state = ClientState.validating
        try:
            model = self.get_model(model_id)  # repository.get_model(model_id)
            fid, infile_name = tempfile.mkstemp(suffix='.h5')
            fod, outfile_name = tempfile.mkstemp(suffix='.h5')
            with open(infile_name, "wb") as fh:
                fh.write(model.getbuffer())

            self.dispatcher.run_cmd("validate {} {}".format(infile_name, outfile_name))

            with open(outfile_name, "r") as fh:
                validation = json.loads(fh.read())

            os.unlink(infile_name)
            os.unlink(outfile_name)

        except Exception as e:
            print("Validation failed with exception {}".format(e), flush=True)
            self.state = ClientState.idle
            return None

        self.state = ClientState.idle
        return validation

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

        self.logs.append(
            "{} {} LOG LEVEL {} MESSAGE {}".format(str(datetime.now()), status.sender.name, status.log_level,
                                                   status.status))
        response = self.connection.SendStatus(status)

    def _send_heartbeat(self, update_frequency=2.0):
        while True:
            heartbeat = alliance.Heartbeat(sender=alliance.Client(name=self.name, role=alliance.WORKER))
            self.connection.SendHeartbeat(heartbeat)
            # self.send_status("HEARTBEAT from {}".format(self.client),log_level=alliance.Status.INFO)
            import time
            time.sleep(update_frequency)

    def run_web(self):
        from flask import Flask
        app = Flask(__name__)

        from .pages import page, style
        @app.route('/')
        def index():
            logs_fancy = str()
            for log in self.logs:
                logs_fancy += "<p>" + log + "</p>\n"

            return page.format(client=self.name, state=ClientStateToString(self.state), style=style, logs=logs_fancy)
            # return {"name": self.name, "State": ClientStateToString(self.state), "Runtime": str(datetime.now() - self.started_at),
            #        "Since": str(self.started_at)}
        import os, sys
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        app.run(host="0.0.0.0", port="8090")
        sys.stdout.close()
        sys.stdout = self._original_stdout

    def run(self):
        import time
        import threading
        threading.Thread(target=self.run_web, daemon=True).start()
        try:
            cnt = 0
            old_state = self.state
            while True:
                time.sleep(1)
                cnt += 1
                if self.state != old_state:
                    print("CLIENT {}".format(ClientStateToString(self.state)), flush=True)
                if cnt > 5:
                    print("CLIENT active",flush=True)
                    cnt = 0
        except KeyboardInterrupt:
            print("ok exiting..")
