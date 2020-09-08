import fedn.common.net.grpc.fedn_pb2 as fedn

import threading



from fedn.clients.reducer.state import ReducerState, ReducerStateToString
from fedn.clients.reducer.interfaces import CombinerInterface, ReducerInferenceInterface
from fedn.clients.reducer.control import ReducerControl

class Reducer:
    def __init__(self, config):
        self.name = config['name']
        self.token = config['token']
        self.control = ReducerControl()
        self.inference = ReducerInferenceInterface()

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
            combiner = CombinerInterface(self, name, address, port)
            self.control.add(combiner)
            ret = {'status': 'added'}
            return jsonify(ret)

        # http://localhost:8090/start?rounds=4&model_id=879fa112-c861-4cb1-a25d-775153e5b548
        @app.route('/start')
        def start():
            timeout = request.args.get('timeout', 180)
            model_id = request.args.get('model_id', '879fa112-c861-4cb1-a25d-775153e5b548')
            rounds = request.args.get('rounds', 1)
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
            combiner_preferred = request.args.get('combiner', None)
            import uuid
            id = str(uuid.uuid4())

            if combiner_preferred:
                combiner = self.control.find(combiner_preferred)
            else:
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
                self.control.set_model_id()
            except fedn.exceptions.ModelError:
                print("Failed to seed control.")

            return result

        #@app.route('/seed')
        #def seed():
        #    try:
        #        result = self.inference.infer(request.args)
        #    except fedn.exceptions.ModelError:
        #        print("no model")
        #
        #    return result


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
