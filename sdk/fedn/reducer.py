import fedn.proto.alliance_pb2 as alliance
import fedn.proto.alliance_pb2_grpc as rpc
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
    def __init__(self, parent, name, address, port, token):
        self.parent = parent
        self.name = name
        self.address = address
        self.port = port
        self.token = token

    def start(self, config):
        channel = grpc.insecure_channel(self.address + ":" + str(self.port))
        control = rpc.ControlStub(channel)
        request = alliance.ControlRequest()
        request.command = alliance.Command.START
        for k, v in config.items():
            p = request.parameter.add()
            p.key = str(k)
            p.value = str(v)

        response = control.Start(request)
        print("Response from combiner {}".format(response.message))


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

    def monitor(self, config):
        self.__state = ReducerState.monitoring

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

    def state(self):
        return self.__state


class Reducer:
    def __init__(self, config):
        self.name = config['name']
        self.token = config['token']
        self.control = ReducerControl()

        # from fedn.algo.fedavg import FEDAVGCombiner
        # self.reducer = FEDAVGCombiner(self.name, self.repository, self)

    def run_web(self):
        from flask import Flask
        from flask import request
        app = Flask(__name__)

        from fedn.web.reducer import page, style
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
            token = request.args.get('token')
            # TODO do validation

            # TODO append and redirect to index.
            combiner = CombinerRepresentation(self, name, address, port, token)
            self.control.add(combiner)
            return "added"

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
        except (KeyboardInterrupt, SystemExit):
            print("Exiting..", flush=True)
