import fedn.common.net.grpc.fedn_pb2 as fedn
import fedn.common.net.grpc.fedn_pb2_grpc as rpc
import grpc
import threading

from enum import Enum


class ReducerState(Enum):
    idle = 1
    instructing = 2
    monitoring = 3


def ReducerStateToString(state):
    if state == ReducerState.idle:
        return "IDLE"
    if state == ReducerState.instructing:
        return "instructing"
    if state == ReducerState.monitoring:
        return "monitoring"

    return "UNKNOWN"


class CombinerRepresentation:
    def __init__(self, parent, name, address, port):
        self.parent = parent
        self.name = name
        self.address = address
        self.port = port

    def start(self, config):
        channel = grpc.insecure_channel(self.address + ":" + str(self.port))
        control = rpc.ControlStub(channel)
        request = fedn.ControlRequest()
        request.command = fedn.Command.START
        for k, v in config.items():
            p = request.parameter.add()
            p.key = str(k)
            p.value = str(v)

        response = control.Start(request)
        print("Response from combiner {}".format(response.message))

    def allowing_clients(self):
        print("Sending message to combiner", flush=True)
        channel = grpc.insecure_channel(self.address + ":" + str(self.port))
        connector = rpc.ConnectorStub(channel)
        request = fedn.ConnectionRequest()
        response = connector.AcceptingClients(request)
        if response.status == fedn.ConnectionStatus.NOT_ACCEPTING:
            print("Sending message to combiner 2", flush=True)
            return False
        if response.status == fedn.ConnectionStatus.ACCEPTING:
            print("Sending message to combiner 3", flush=True)
            return True
        if response.status == fedn.ConnectionStatus.TRY_AGAIN_LATER:
            print("Sending message to combiner 4", flush=True)
            return False

        print("Sending message to combiner 5??", flush=True)
        return False


class ReducerControl:

    def __init__(self):
        self.__state = ReducerState.idle
        self.combiners = []

    def instruct(self, config):
        if self.__state == ReducerState.instructing:
            print("Already set in INSTRUCTING state", flush=True)
            return

        self.__state = ReducerState.instructing

        print("REDUCER: STARTING COMBINERS", flush=True)
        for combiner in self.combiners:
            print("REDUCER: STARTING {}".format(combiner.name), flush=True)
            combiner.start(config)
        print("REDUCER: STARTED {} COMBINERS".format(len(self.combiners), flush=True))

        self.__state = ReducerState.monitoring

    def monitor(self, config=None):
        if self.__state == ReducerState.monitoring:
            print("monitoring")
        # todo connect to combiners and listen for globalmodelupdate request.
        # use the globalmodel received to start the reducer combiner method on received models to construct its own model.

    def add(self, combiner):
        if self.__state != ReducerState.idle:
            print("Reducer is not idle, cannot add additional combiner")
            return
        if self.find(combiner.name):
            return
        print("adding combiner {}".format(combiner.name), flush=True)
        self.combiners.append(combiner)

    def remove(self, combiner):
        if self.__state != ReducerState.idle:
            print("Reducer is not idle, cannot remove combiner")
            return
        self.combiners.remove(combiner)

    def find(self, name):
        for combiner in self.combiners:
            if name == combiner.name:
                return combiner
        return None

    def find_available_combiner(self):
        for combiner in self.combiners:
            if combiner.allowing_clients():
                return combiner
        return None

    def state(self):
        return self.__state


class ReducerInference:
    def __init__(self):
        self.model_wrapper = None

    def set(self, model):
        self.model_wrapper = model

    def infer(self, params):
        results = None
        if self.model_wrapper:
            results = self.model_wrapper.infer(params)

        return results




class Reducer:
    def __init__(self, config):
        self.name = config['name']
        self.token = config['token']
        self.control = ReducerControl()
        self.inference = ReducerInference()

        # from fedn.algo.fedavg import FEDAVGCombiner
        # self.reducer = FEDAVGCombiner(self.name, self.repository, self)

    def run_web(self):
        from flask import Flask
        from flask import request, jsonify
        app = Flask(__name__)

        from fedn.common.net.web.reducer import page, style
        @app.route('/')
        def index():
            # logs_fancy = str()
            # for log in self.logs:
            #    logs_fancy += "<p>" + log + "</p>\n"

            return page.format(client=self.name, state=ReducerStateToString(self.control.state()), style=style,
                               logs=None)
            # return {"name": self.name, "State": ClientStateToString(self.state), "Runtime": str(datetime.now() - self.started_at),
            #        "Since": str(self.started_at)}

        # http://localhost:8090/add?name=combiner&address=combiner&port=12080&token=e9a3cb4c5eaff546eec33ff68a7fbe232b68a192
        @app.route('/add')
        def add():
            # TODO check for get variables
            name = request.args.get('name')
            address = request.args.get('address')
            port = request.args.get('port')
            # token = request.args.get('token')
            # TODO do validation

            # TODO append and redirect to index.
            combiner = CombinerRepresentation(self, name, address, port)
            self.control.add(combiner)
            ret = {'status': 'added'}
            return jsonify(ret)

        # http://localhost:8090/start?rounds=4&model_id=879fa112-c861-4cb1-a25d-775153e5b548
        @app.route('/start')
        def start():
            timeout = request.args.get('timeout', 180)
            model_id = request.args.get('model_id', '879fa112-c861-4cb1-a25d-775153e5b548')
            rounds = request.args.get('rounds', 3)
            active_clients = request.args.get('active_clients', 2)
            clients_required = request.args.get('clients_required', 2)
            clients_requested = request.args.get('clients_requested', 2)

            config = {'round_timeout': timeout, 'model_id': model_id, 'rounds': rounds,
                      'active_clients': active_clients, 'clients_required': clients_required,
                      'clients_requested': clients_requested}

            self.control.instruct(config)
            return "started"

        from flask import jsonify, abort
        @app.route('/assign')
        def assign():
            name = request.args.get('name', None)
            import uuid
            id = str(uuid.uuid4())

            combiner = self.control.find_available_combiner()
            if combiner:
                response = {'host': combiner.address, 'port': combiner.port}
                return jsonify(response)
            elif combiner is None:
                abort(404, description="Resource not found")
            # 1.receive client parameters
            # 2. check with available combiners if any clients are needed
            # 3. let client know where to connect.
            return

        @app.route('/infer')
        def infer():
            result = ""
            try:
                result = self.inference.infer(request.args)
            except fedn.exceptions.ModelError:
                print("no model")

            return result

        # import os, sys
        # self._original_stdout = sys.stdout
        # sys.stdout = open(os.devnull, 'w')
        app.run(host="0.0.0.0", port="8090")
        # sys.stdout.close()
        # sys.stdout = self._original_stdout

    def run(self):

        threading.Thread(target=self.run_web, daemon=True).start()

        import time
        try:
            while True:
                time.sleep(1)
                print("Reducer in {} state".format(ReducerStateToString(self.control.state())), flush=True)
                self.control.monitor()
        except (KeyboardInterrupt, SystemExit):
            print("Exiting..", flush=True)
