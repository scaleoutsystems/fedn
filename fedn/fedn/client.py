import json
import os
import sys
import io
import uuid
import tempfile
import threading, queue
import time
from aiohttp import client

import grpc

import fedn.common.net.grpc.fedn_pb2 as fedn
import fedn.common.net.grpc.fedn_pb2_grpc as rpc
from fedn.common.net.connect import ConnectorClient, Status
from fedn.common.control.package import PackageRuntime

from fedn.utils.logger import Logger
from fedn.utils.helpers import get_helper

# TODO Remove from this level. Abstract to unified non implementation specific client.
from fedn.utils.dispatcher import Dispatcher

from fedn.clients.client.state import ClientState, ClientStateToString

CHUNK_SIZE = 1024 * 1024

from datetime import datetime


class Client:
    """FEDn Client. Service running on client/datanodes in a federation,
       recieving and handling model update and model validation requests.
    
    Attibutes
    ---------
    config: dict
        A configuration dictionary containing connection information for
        the discovery service (controller) and settings governing e.g. 
        client-combiner assignment behavior.
    
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config: dict
            A configuration dictionary containing connection information for
            the discovery service (controller) and settings governing e.g. 
            client-combiner assignment behavior.
        """

        self.state = None
        self.error_state = False
        self._attached = False
        self._missed_heartbeat=0
        self.config  = config

        self.connector = ConnectorClient(config['discover_host'],
                                         config['discover_port'],
                                         config['token'],
                                         config['name'],
                                         config['remote_compute_context'],
                                         config['preferred_combiner'],
                                         config['client_id'],
                                         secure=config['secure'],
                                         preshared_cert=config['preshared_cert'],
                                         verify_cert=config['verify_cert'])
                                         
        self.name = config['name']
        dirname = time.strftime("%Y%m%d-%H%M%S")
        self.run_path = os.path.join(os.getcwd(), dirname)
        os.mkdir(self.run_path)

        self.logger = Logger(to_file=config['logfile'], file_path=self.run_path)
        self.started_at = datetime.now()
        self.logs = []

        self.inbox = queue.Queue()

        # Attach to the FEDn network (get combiner)
        client_config = self._attach()
     
        self._initialize_dispatcher(config)

        self._initialize_helper(client_config)
        if not self.helper:
            print("Failed to retrive helper class settings! {}".format(client_config), flush=True)

        self._subscribe_to_combiner(config)

        self.state = ClientState.idle

    def _detach(self):
        # Setting _attached to False will make all processing threads return 
        if not self._attached:
            print("Client is not attached.",flush=True)

        self._attached = False
        # Close gRPC connection to combiner
        self._disconnect()

    def _attach(self):
        """ """
        # Ask controller for a combiner and connect to that combiner.
        if self._attached: 
            print("Client is already attached. ",flush=True)
            return None

        client_config = self._assign()
        self._connect(client_config)

        if client_config: 
            self._attached=True
        return client_config

    def _initialize_helper(self,client_config):
        
        if 'model_type' in client_config.keys():
            self.helper = get_helper(client_config['model_type'])

    def _subscribe_to_combiner(self,config):
        """Listen to combiner message stream and start all processing threads. 
        
        """

        # Start sending heartbeats to the combiner. 
        threading.Thread(target=self._send_heartbeat, kwargs={'update_frequency': config['heartbeat_interval']}, daemon=True).start()

        # Start listening for combiner training and validation messages 
        if config['trainer'] == True:
            threading.Thread(target=self._listen_to_model_update_request_stream, daemon=True).start()
        if config['validator'] == True:
            threading.Thread(target=self._listen_to_model_validation_request_stream, daemon=True).start()
        self._attached = True

        # Start processing the client message inbox
        threading.Thread(target=self.process_request, daemon=True).start()

    def _initialize_dispatcher(self, config):
        """ """
        if config['remote_compute_context']:
            pr = PackageRuntime(os.getcwd(), os.getcwd())

            retval = None
            tries = 10

            while tries > 0:
                retval = pr.download(config['discover_host'], config['discover_port'], config['token'])
                if retval:
                    break
                time.sleep(60)
                print("No compute package available... retrying in 60s Trying {} more times.".format(tries), flush=True)
                tries -= 1

            if retval:
                if not 'checksum' in config:
                    print(
                        "\nWARNING: Skipping security validation of local package!, make sure you trust the package source.\n",
                        flush=True)
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
            from_path = os.path.join(os.getcwd(), 'client')

            from distutils.dir_util import copy_tree
            copy_tree(from_path, self.run_path)
            self.dispatcher = Dispatcher(dispatch_config, self.run_path)


    def  _assign(self):
        """Contacts the controller and asks for combiner assignment. """

        print("Asking for assignment!", flush=True)
        while True:
            status, response = self.connector.assign()
            if status == Status.TryAgain:
                print(response, flush=True)
                time.sleep(5)
                continue
            if status == Status.Assigned:
                client_config = response
                break
            if status == Status.UnAuthorized:
                print(response, flush=True)
                sys.exit("Exiting: Unauthorized")
            if status == Status.UnMatchedConfig:
                print(response, flush=True)
                sys.exit("Exiting: UnMatchedConfig")
            time.sleep(5)
            print(".", end=' ', flush=True)
        
        print("Got assigned!", flush=True)
        return client_config

    def _connect(self, client_config):
        """Connect to assigned combiner. 
        
        Parameters
        ----------
        client_config : dict
            A dictionary with connection information and settings
            for the assigned combiner. 
        
        """

        # TODO use the client_config['certificate'] for setting up secure comms'
        if client_config['certificate']:
            import base64
            cert = base64.b64decode(client_config['certificate'])  # .decode('utf-8')
            credentials = grpc.ssl_channel_credentials(root_certificates=cert)
            channel = grpc.secure_channel("{}:{}".format(client_config['host'], str(client_config['port'])),
                                          credentials)
        else:
            channel = grpc.insecure_channel("{}:{}".format(client_config['host'], str(client_config['port'])))

        self.channel = channel

        self.connection = rpc.ConnectorStub(channel)
        self.orchestrator = rpc.CombinerStub(channel)
        self.models = rpc.ModelServiceStub(channel)

        print("Client: {} connected {} to {}:{}".format(self.name,
                                                        "SECURED" if client_config['certificate'] else "INSECURE",
                                                        client_config['host'], client_config['port']), flush=True)
        
        print("Client: Using {} compute package.".format(client_config["package"]))

    def _disconnect(self):
        self.channel.close()

    def get_model(self, id):
        """Fetch a model from the assigned combiner. 

        Downloads the model update object via a gRPC streaming channel, Dowload. 
        
        Parameters
        ----------
        id : str
            The id of the model update object. 
        
        """

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
        """Send a model update to the assigned combiner. 

        Uploads the model updated object via a gRPC streaming channel, Upload. 

        Parameters
        ----------
        model : BytesIO, object
            The  model update object. 
        id : str
            The id of the model update object.
        """       

        from io import BytesIO

        if not isinstance(model, BytesIO):
            bt = BytesIO()

            for d in model.stream(32 * 1024):
                bt.write(d)
        else:
            bt = model

        bt.seek(0, 0)

        def upload_request_generator(mdl):
            """

            :param mdl:
            """
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

    def _listen_to_model_update_request_stream(self):
        """Subscribe to the model update request stream. """

        r = fedn.ClientAvailableMessage()
        r.sender.name = self.name
        r.sender.role = fedn.WORKER
        metadata = [('client', r.sender.name)]
        _disconnect = False

        while True:
            try:
                for request in self.orchestrator.ModelUpdateRequestStream(r, metadata=metadata):
                    if request.sender.role == fedn.COMBINER:
                        # Process training request
                        self._send_status("Received model update request.", log_level=fedn.Status.AUDIT,
                                         type=fedn.StatusType.MODEL_UPDATE_REQUEST, request=request)

                        self.inbox.put(('train', request))
                    
                    if not self._attached: 
                        return 

            except grpc.RpcError as e:
                status_code = e.code()
                #TODO: make configurable
                timeout = 5
                #print("CLIENT __listen_to_model_update_request_stream: GRPC ERROR {} retrying in {}..".format(
                #    status_code.name, timeout), flush=True)
                time.sleep(timeout) 
            except:
                raise

            if not self._attached: 
                return

    def _listen_to_model_validation_request_stream(self):
        """Subscribe to the model validation request stream. """

        r = fedn.ClientAvailableMessage()
        r.sender.name = self.name
        r.sender.role = fedn.WORKER
        while True:
            try:
                for request in self.orchestrator.ModelValidationRequestStream(r):
                    # Process validation request
                    model_id = request.model_id
                    self._send_status("Recieved model validation request.", log_level=fedn.Status.AUDIT,
                                     type=fedn.StatusType.MODEL_VALIDATION_REQUEST, request=request)
                    self.inbox.put(('validate', request))

            except grpc.RpcError as e:
                status_code = e.code()
                # TODO: make configurable
                timeout = 5
                #print("CLIENT __listen_to_model_validation_request_stream: GRPC ERROR {} retrying in {}..".format(
                #    status_code.name, timeout), flush=True)
                time.sleep(timeout)
            except:
                raise 

            if not self._attached: 
                return

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
                    model_id, meta = self._process_training_request(request.model_id)
                    processing_time = time.time()-tic
                    meta['processing_time'] = processing_time

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
                    metrics = self._process_validation_request(request.model_id)

                    if metrics != None:
                        # Send validation
                        validation = fedn.ModelValidation()
                        validation.sender.name = self.name
                        validation.sender.role = fedn.WORKER
                        validation.receiver.name = request.sender.name
                        validation.receiver.role = request.sender.role
                        validation.model_id = str(request.model_id)
                        validation.data = json.dumps(metrics)
                        self.str = str(datetime.now())
                        validation.timestamp = self.str
                        validation.correlation_id = request.correlation_id
                        response = self.orchestrator.SendModelValidation(validation)
                        self._send_status("Model validation completed.", log_level=fedn.Status.AUDIT,
                                            type=fedn.StatusType.MODEL_VALIDATION, request=validation)
                    else:
                        self._send_status("Client {} failed to complete model validation.".format(self.name),
                                            log_level=fedn.Status.WARNING, request=request)

                    self.state = ClientState.idle
                    self.inbox.task_done()
            except queue.Empty:
                pass

    def _process_training_request(self, model_id):
        """Process a training (model update) request. 
        
        Parameters
        ----------
        model_id : Str
            The id of the model to update.
        
        """

        self._send_status("\t Starting processing of training request for model_id {}".format(model_id))
        self.state = ClientState.training

        try:
            meta = {}
            tic = time.time()
            mdl = self.get_model(str(model_id))
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

            # Push model update to combiner server
            updated_model_id = uuid.uuid4()
            self.set_model(out_model, str(updated_model_id))
            meta['upload_model'] = time.time() - tic

            os.unlink(inpath)
            os.unlink(outpath)

        except Exception as e:
            print("ERROR could not process training request due to error: {}".format(e), flush=True)
            updated_model_id = None
            meta = {'status': 'failed', 'error': str(e)}

        self.state = ClientState.idle

        return updated_model_id, meta

    def _process_validation_request(self, model_id):
        self._send_status("Processing validation request for model_id {}".format(model_id))
        self.state = ClientState.validating
        try:
            model = self.get_model(str(model_id))
            inpath = self.helper.get_tmp_path()

            with open(inpath, "wb") as fh:
                fh.write(model.getbuffer())

            _, outpath = tempfile.mkstemp()
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

    def _handle_combiner_failure(self):
        """ Register failed combiner connection. 

        """
        self._missed_heartbeat += 1 
        if self._missed_heartbeat > self.config['reconnect_after_missed_heartbeat']: 
            self._detach()

    def _send_heartbeat(self, update_frequency=2.0):
        """Send a heartbeat to the combiner. 
        
        Parameters
        ----------
        update_frequency : float
            The interval in seconds between heartbeat messages.
        
        """

        while True:
            heartbeat = fedn.Heartbeat(sender=fedn.Client(name=self.name, role=fedn.WORKER))
            try:
                self.connection.SendHeartbeat(heartbeat)
                self._missed_heartbeat = 0
            except grpc.RpcError as e:
                status_code = e.code()
                print("CLIENT heartbeat: GRPC ERROR {} retrying..".format(status_code.name), flush=True)
                self._handle_combiner_failure()

            time.sleep(update_frequency)
            if not self._attached: 
                return 

    def _send_status(self, msg, log_level=fedn.Status.INFO, type=None, request=None):
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


    def run_web(self):
        """Starts a local logging UI (Flask app) serving on port 8080. 
        
        Currently not in use as default. 
        
        """
        from flask import Flask
        app = Flask(__name__)

        from fedn.common.net.web.client import page, style
        @app.route('/')
        def index():
            """

            :return:
            """
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
        """ Main run loop. """
        #threading.Thread(target=self.run_web, daemon=True).start()
        try:
            cnt = 0
            old_state = self.state
            while True:
                time.sleep(1)
                cnt += 1
                if self.state != old_state:
                    print("{}:CLIENT in {} state".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ClientStateToString(self.state)), flush=True)
                if cnt > 5:
                    print("{}:CLIENT active".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), flush=True)
                    cnt = 0
                if not self._attached:
                    print("Detatched from combiner.", flush=True)
                    # TODO: Implement a check/condition to ulitmately close down if too many reattachment attepts have failed. s  
                    self._attach()
                    self._subscribe_to_combiner(self.config)
                if self.error_state:
                    return
        except KeyboardInterrupt:
            print("Ok, exiting..")
