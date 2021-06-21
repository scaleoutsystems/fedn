import json
import os
import tempfile
import threading
import time

import fedn.common.net.grpc.fedn_pb2 as fedn
import fedn.common.net.grpc.fedn_pb2_grpc as rpc
import grpc
# TODO Remove from this level. Abstract to unified non implementation specific client.
from fedn.utils.dispatcher import Dispatcher

CHUNK_SIZE = 1024 * 1024

from datetime import datetime

from fedn.clients.client.state import ClientState, ClientStateToString

from fedn.utils.helpers import get_helper

class Client:
    """FEDn Client. """

    def __init__(self, config):

        self.state = None
        self.error_state = False
        from fedn.common.net.connect import ConnectorClient, Status
        self.connector = ConnectorClient(config['discover_host'],
                                         config['discover_port'],
                                         config['token'],
                                         config['name'],
                                         config['preferred_combiner'],
                                         config['client_id'],
                                         secure=config['secure'],
                                         preshared_cert=config['preshared_cert'],
                                         verify_cert=config['verify_cert'])
        self.name = config['name']
        import time
        dirname = time.strftime("%Y%m%d-%H%M%S")
        self.run_path = os.path.join(os.getcwd(), dirname)
        os.mkdir(self.run_path)

        from fedn.utils.logger import Logger
        self.logger = Logger(to_file=config['logfile'],file_path=self.run_path)
        self.started_at = datetime.now()
        self.logs = []
        client_config = {}
        print("Asking for assignment",flush=True)
        import time
        while True:
            status, response = self.connector.assign()
            if status == Status.TryAgain:
                time.sleep(5)
                continue
            if status == Status.Assigned:
                client_config = response
                break
            time.sleep(5)
            print(".", end=' ', flush=True)

        print("Got assigned!", flush=True)

        # TODO use the client_config['certificate'] for setting up secure comms'
        if client_config['certificate']:
            import base64
            cert = base64.b64decode(client_config['certificate'])  # .decode('utf-8')
            credentials = grpc.ssl_channel_credentials(root_certificates=cert)
            channel = grpc.secure_channel("{}:{}".format(client_config['host'], str(client_config['port'])),
                                          credentials)
        else:
            channel = grpc.insecure_channel("{}:{}".format(client_config['host'], str(client_config['port'])))

        self.connection = rpc.ConnectorStub(channel)
        self.orchestrator = rpc.CombinerStub(channel)
        self.models = rpc.ModelServiceStub(channel)

        print("Client: {} connected {} to {}:{}".format(self.name,
                                                        "SECURED" if client_config['certificate'] else "INSECURE",
                                                        client_config['host'], client_config['port']), flush=True)
        if config['remote_compute_context']:
            from fedn.common.control.package import PackageRuntime
            pr = PackageRuntime(os.getcwd(), os.getcwd())

            retval = None
            tries = 10

            while tries > 0:
                retval =  pr.download(config['discover_host'], config['discover_port'], config['token'])
                if retval:
                    break
                time.sleep(60)
                print("No compute package available... retrying in 60s Trying {} more times.".format(tries),flush=True)
                tries -= 1

            if retval:
                if not 'checksum' in config:
                    print("\nWARNING: Skipping security validation of local package!, make sure you trust the package source.\n",flush=True)
                else:
                    checks_out = pr.validate(config['checksum'])
                    if not checks_out:
                        print("Validation was enforced and invalid, client closing!")
                        self.error_state = True
                        return

            if retval:
                pr.unpack()

            self.dispatcher = pr.dispatcher(self.run_path)
            try:
                print("Running Dispatcher for entrypoint: startup", flush=True)
                self.dispatcher.run_cmd("startup")
            except KeyError:
                pass
        else:
            # TODO: Deprecate
            dispatch_config = {'entry_points':
                                   {'predict': {'command': 'python3 predict.py'},
                                    'train': {'command': 'python3 train.py'},
                                    'validate': {'command': 'python3 validate.py'}}}
            dispatch_dir = os.getcwd()
            from_path = os.path.join(os.getcwd(),'client')

            from distutils.dir_util import copy_tree
            copy_tree(from_path, run_path)
            self.dispatcher = Dispatcher(dispatch_config, self.run_path)

        self.lock = threading.Lock()

        if 'model_type' in client_config.keys():
            self.helper = get_helper(client_config['model_type'])

        if not self.helper:
            print("Failed to retrive helper class settings! {}".format(client_config),flush=True)

        threading.Thread(target=self._send_heartbeat, daemon=True).start()
        threading.Thread(target=self.__listen_to_model_update_request_stream, daemon=True).start()
        threading.Thread(target=self.__listen_to_model_validation_request_stream, daemon=True).start()

        self.state = ClientState.idle

    def get_model(self, id):
        """Fetch model from the Combiner. """

        from io import BytesIO
        data = BytesIO()

        for part in self.models.Download(fedn.ModelRequest(id=id)):

            if part.status == fedn.ModelStatus.IN_PROGRESS:
                data.write(part.data)

            if part.status == fedn.ModelStatus.OK:
                return data

            if part.status == fedn.ModelStatus.FAILED:
                return None

        return data

    def set_model(self, model, id):
        """Upload a model to the Combiner. """

        from io import BytesIO

        if not isinstance(model, BytesIO):
            bt = BytesIO()

            for d in model.stream(32 * 1024):
                bt.write(d)
        else:
            bt = model

        bt.seek(0, 0)

        def upload_request_generator(mdl):
            i = 1
            while True:
                b = mdl.read(CHUNK_SIZE)
                if b:
                    result = fedn.ModelRequest(data=b, id=id, status=fedn.ModelStatus.IN_PROGRESS)
                else:
                    result = fedn.ModelRequest(id=id, status=fedn.ModelStatus.OK)

                yield result
                if not b:
                    break

        result = self.models.Upload(upload_request_generator(bt))

        return result

    def __listen_to_model_update_request_stream(self):
        """Subscribe to the model update request stream. """
        r = fedn.ClientAvailableMessage()
        r.sender.name = self.name
        r.sender.role = fedn.WORKER
        metadata = [('client', r.sender.name)]
        import time
        while True:
            try:
                for request in self.orchestrator.ModelUpdateRequestStream(r, metadata=metadata):
                    if request.sender.role == fedn.COMBINER:
                        # Process training request
                        global_model_id = request.model_id
                        # TODO: Error handling
                        self.send_status("Received model update request.", log_level=fedn.Status.AUDIT,
                                         type=fedn.StatusType.MODEL_UPDATE_REQUEST, request=request)

                        tic = time.time()
                        model_id, meta = self.__process_training_request(global_model_id)
                        processing_time = time.time()-tic
                        meta['processing_time'] = processing_time
                        print(meta,flush=True)

                        if model_id != None:
                            # Notify the combiner that a model update is available
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
                            #TODO: Check responses
                            response = self.orchestrator.SendModelUpdate(update)

                            self.send_status("Model update completed.", log_level=fedn.Status.AUDIT,
                                             type=fedn.StatusType.MODEL_UPDATE, request=update)

                        else:
                            self.send_status("Client {} failed to complete model update.",
                                             log_level=fedn.Status.WARNING,
                                             request=request)
            except grpc.RpcError as e:
                status_code = e.code()
                timeout = 5
                print("CLIENT __listen_to_model_update_request_stream: GRPC ERROR {} retrying in {}..".format(
                    status_code.name, timeout), flush=True)
                import time
                time.sleep(timeout)

    def __listen_to_model_validation_request_stream(self):
        """Subscribe to the model validation request stream. """
        r = fedn.ClientAvailableMessage()
        r.sender.name = self.name
        r.sender.role = fedn.WORKER
        while True:
            try:
                for request in self.orchestrator.ModelValidationRequestStream(r):
                    # Process training request
                    model_id = request.model_id
                    # TODO: Error handling
                    self.send_status("Recieved model validation request.", log_level=fedn.Status.AUDIT,
                                     type=fedn.StatusType.MODEL_VALIDATION_REQUEST, request=request)
                    metrics = self.__process_validation_request(model_id)

                    if metrics != None:
                        # Send validation
                        validation = fedn.ModelValidation()
                        validation.sender.name = self.name
                        validation.sender.role = fedn.WORKER
                        validation.receiver.name = request.sender.name
                        validation.receiver.role = request.sender.role
                        validation.model_id = str(model_id)
                        validation.data = json.dumps(metrics)
                        self.str = str(datetime.now())
                        validation.timestamp = self.str
                        validation.correlation_id = request.correlation_id
                        response = self.orchestrator.SendModelValidation(validation)
                        self.send_status("Model validation completed.", log_level=fedn.Status.AUDIT,
                                         type=fedn.StatusType.MODEL_VALIDATION, request=validation)
                    else:
                        self.send_status("Client {} failed to complete model validation.".format(self.name),
                                         log_level=fedn.Status.WARNING, request=request)
            except grpc.RpcError as e:
                status_code = e.code()
                timeout = 5
                print("CLIENT __listen_to_model_validation_request_stream: GRPC ERROR {} retrying in {}..".format(
                    status_code.name, timeout), flush=True)
                import time
                time.sleep(timeout)

    def __process_training_request(self, model_id):

        self.send_status("\t Starting processing of training request for model_id {}".format(model_id))
        self.state = ClientState.training

        try:
            meta = {}
            tic = time.time()
            mdl = self.get_model(str(model_id))
            meta['fetch_model'] = time.time()-tic

            import sys
            inpath = self.helper.get_tmp_path()
            with open(inpath,'wb') as fh:
                fh.write(mdl.getbuffer())

            outpath = self.helper.get_tmp_path()
            tic = time.time()
            #TODO: Check return status, fail gracefully
            self.dispatcher.run_cmd("train {} {}".format(inpath, outpath))
            meta['exec_training'] = time.time()-tic

            tic = time.time()
            import io
            out_model = None
            with open(outpath, "rb") as fr:
                out_model = io.BytesIO(fr.read())

            import uuid
            updated_model_id = uuid.uuid4()
            self.set_model(out_model, str(updated_model_id))
            meta['upload_model'] = time.time()-tic

            os.unlink(inpath)
            os.unlink(outpath)

        except Exception as e:
            print("ERROR could not process training request due to error: {}".format(e),flush=True)
            updated_model_id = None
            meta = {'status':'failed','error':str(e)}

        self.state = ClientState.idle

        return updated_model_id, meta 

    def __process_validation_request(self, model_id):
        self.send_status("Processing validation request for model_id {}".format(model_id))
        self.state = ClientState.validating
        try:
            model = self.get_model(str(model_id))
            inpath = self.helper.get_tmp_path()

            with open(inpath, "wb") as fh:
                fh.write(model.getbuffer())

            _,outpath = tempfile.mkstemp()
            self.dispatcher.run_cmd("validate {} {}".format(inpath, outpath))

            with open(outpath, "r") as fh:
                validation = json.loads(fh.read())

            os.unlink(inpath)
            os.unlink(outpath)

        except Exception as e:
            print("Validation failed with exception {}".format(e), flush=True)
            raise
            self.state = ClientState.idle
            return None

        self.state = ClientState.idle
        return validation

    def send_status(self, msg, log_level=fedn.Status.INFO, type=None, request=None):
        """Send status message. """

        from google.protobuf.json_format import MessageToJson
        
        status = fedn.Status()
        status.timestamp = str(datetime.now())
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
        response = self.connection.SendStatus(status)

    def _send_heartbeat(self, update_frequency=2.0):
        """Send a heartbeat to the Combiner. """
        while True:
            heartbeat = fedn.Heartbeat(sender=fedn.Client(name=self.name, role=fedn.WORKER))
            try:
                self.connection.SendHeartbeat(heartbeat)
            except grpc.RpcError as e:
                status_code = e.code()
                print("CLIENT heartbeat: GRPC ERROR {} retrying..".format(status_code.name), flush=True)
            import time
            time.sleep(update_frequency)

    def run_web(self):
        from flask import Flask
        app = Flask(__name__)

        from fedn.common.net.web.client import page, style
        @app.route('/')
        def index():
            logs_fancy = str()
            for log in self.logs:
                logs_fancy += "<p>" + log + "</p>\n"

            return page.format(client=self.name, state=ClientStateToString(self.state), style=style, logs=logs_fancy)

        import os, sys
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        app.run(host="0.0.0.0", port="8080")
        sys.stdout.close()
        sys.stdout = self._original_stdout

    def run(self):
        import time
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
                    print("CLIENT active", flush=True)
                    cnt = 0
                if self.error_state:
                    return
        except KeyboardInterrupt:
            print("ok exiting..")
