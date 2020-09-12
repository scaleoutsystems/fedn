from fedn.clients.reducer.interfaces import CombinerInterface
from fedn.clients.reducer.state import ReducerStateToString
from flask import Flask
from flask import jsonify, abort
from flask import render_template
from flask import request
from flask_wtf.csrf import CSRFProtect


class ReducerRestService:
    def __init__(self, name, control, certificate=None):
        self.name = name
        self.control = control
        self.certificate = certificate

    def run(self):
        app = Flask(__name__)
        csrf = CSRFProtect()
        import os
        SECRET_KEY = os.urandom(32)
        app.config['SECRET_KEY'] = SECRET_KEY

        csrf.init_app(app)

        @app.route('/')
        def index():
            # logs_fancy = str()
            # for log in self.logs:
            #    logs_fancy += "<p>" + log + "</p>\n"
            client = self.name
            state = ReducerStateToString(self.control.state())
            logs = None
            refresh = True
            return render_template('index.html', client=client, state=state, logs=logs, refresh=refresh)

        # http://localhost:8090/add?name=combiner&address=combiner&port=12080&token=e9a3cb4c5eaff546eec33ff68a7fbe232b68a192
        @app.route('/add')
        def add():
            # TODO check for get variables
            name = request.args.get('name', None)
            address = request.args.get('address', None)
            port = request.args.get('port', None)
            # token = request.args.get('token')
            # TODO do validation

            if port is None or address is None or name is None:
                return "Please specify correct parameters."

            # TODO append and redirect to index.
            combiner = CombinerInterface(self, name, address, port)
            self.control.add(combiner)
            ret = {'status': 'added'}
            return jsonify(ret)

        # http://localhost:8090/start?rounds=4&model_id=879fa112-c861-4cb1-a25d-775153e5b548
        @app.route('/start', methods=['GET', 'POST'])
        def start():

            if request.method == 'POST':
                timeout = request.form.get('timeout', 180)
                model_id = request.form.get('model_id', '879fa112-c861-4cb1-a25d-775153e5b548')
                rounds = int(request.form.get('rounds', 1))
                task = (request.form.get('task', ''))
                active_clients = request.form.get('active_clients', 2)
                clients_required = request.form.get('clients_required', 2)
                clients_requested = request.form.get('clients_requested', 2)

                config = {'round_timeout': timeout, 'model_id': model_id, 'rounds': rounds,
                          'active_clients': active_clients, 'clients_required': clients_required,
                          'clients_requested': clients_requested, 'task': task}

                self.control.instruct(config)
                from flask import redirect, url_for
                return redirect(url_for('index', message="Sent execution plan."))

            client = self.name
            state = ReducerStateToString(self.control.state())
            logs = None
            refresh = False
            return render_template('index.html', client=client, state=state, logs=logs, refresh=refresh)

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

        # @app.route('/seed')
        # def seed():
        #    try:
        #        result = self.inference.infer(request.args)
        #    except fedn.exceptions.ModelError:
        #        print("no model")
        #
        #    return result

        # import os, sys
        # self._original_stdout = sys.stdout
        # sys.stdout = open(os.devnull, 'w')
        if self.certificate:
            print("trying to connect with certs {} and key {}".format(str(self.certificate.cert_path), str(self.certificate.key_path)), flush =True)
            app.run(host="0.0.0.0", port="8090", ssl_context=(str(self.certificate.cert_path), str(self.certificate.key_path)))
        #secure = False
        #secure_adhoc = False

        #if secure and secure_adhoc:
        #    app.run(host="0.0.0.0", port="8090", ssl_context='adhoc')
        #elif secure:
        #    app.run(host="0.0.0.0", port="8090", ssl_context=('cert.pem', 'key.pem'))
        #else:
        #    app.run(host="0.0.0.0", port="8090")
        # sys.stdout.close()
        # sys.stdout = self._original_stdout
