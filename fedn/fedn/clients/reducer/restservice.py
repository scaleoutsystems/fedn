from fedn.clients.reducer.interfaces import CombinerInterface
from fedn.clients.reducer.state import ReducerState, ReducerStateToString
from flask import Flask, flash
from flask import jsonify, abort
from flask import render_template
from flask import request, redirect, url_for
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = {'gz', 'bz2', 'tar', 'zip'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class ReducerRestService:
    def __init__(self, name, control, certificate_manager, certificate=None):
        self.name = name
        self.control = control
        self.certificate = certificate
        self.certificate_manager = certificate_manager

        self.current_compute_context = self.control.get_compute_context()

    def run(self):
        app = Flask(__name__)
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
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
            if self.control.state() == ReducerState.setup:
                return render_template('setup.html', client=client, state=state, logs=logs, refresh=refresh,
                                       dashboardhost=os.environ["FEDN_DASHBOARD_HOST"],
                                       dashboardport=os.environ["FEDN_DASHBOARD_PORT"])

            return render_template('index.html', client=client, state=state, logs=logs, refresh=refresh,
                                   dashboardhost=os.environ["FEDN_DASHBOARD_HOST"],
                                   dashboardport=os.environ["FEDN_DASHBOARD_PORT"])

        # http://localhost:8090/add?name=combiner&address=combiner&port=12080&token=e9a3cb4c5eaff546eec33ff68a7fbe232b68a192
        @app.route('/add')
        def add():
            if self.control.state() == ReducerState.setup:
                return "Error, not configured"
            # TODO check for get variables
            name = request.args.get('name', None)
            address = request.args.get('address', None)
            port = request.args.get('port', None)
            # token = request.args.get('token')
            # TODO do validation

            if port is None or address is None or name is None:
                return "Please specify correct parameters."

            certificate, key = self.certificate_manager.get_or_create(address).get_keypair_raw()
            import base64
            cert_b64 = base64.b64encode(certificate)
            key_b64 = base64.b64encode(key)

            # TODO append and redirect to index.
            import copy
            combiner = CombinerInterface(self, name, address, port, copy.deepcopy(certificate), copy.deepcopy(key))
            self.control.add(combiner)

            ret = {'status': 'added', 'certificate': str(cert_b64).split('\'')[1],
                   'key': str(key_b64).split('\'')[1]}  # TODO remove ugly string hack
            return jsonify(ret)

        # http://localhost:8090/start?rounds=4&model_id=879fa112-c861-4cb1-a25d-775153e5b548
        @app.route('/start', methods=['GET', 'POST'])
        def start():
            if self.control.state() == ReducerState.setup:
                return "Error, not configured"

            if request.method == 'POST':
                timeout = request.form.get('timeout', 180)
                model_id = request.form.get('model', '879fa112-c861-4cb1-a25d-775153e5b548')
                if model_id == '':
                    model_id = '879fa112-c861-4cb1-a25d-775153e5b548'
                rounds = int(request.form.get('rounds', 1))
                task = (request.form.get('task', ''))
                active_clients = request.form.get('active_clients', 2)
                clients_required = request.form.get('clients_required', 2)
                clients_requested = request.form.get('clients_requested', 8)

                config = {'round_timeout': timeout, 'model_id': model_id, 'rounds': rounds,
                          'active_clients': active_clients, 'clients_required': clients_required,
                          'clients_requested': clients_requested, 'task': task}

                config['model_id'] = '879fa112-c861-4cb1-a25d-775153e5b548'
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
            if self.control.state() == ReducerState.setup:
                return "Error, not configured"
            name = request.args.get('name', None)
            combiner_preferred = request.args.get('combiner', None)
            import uuid
            id = str(uuid.uuid4())

            if combiner_preferred:
                combiner = self.control.find(combiner_preferred)
            else:
                combiner = self.control.find_available_combiner()

            if combiner:
                # certificate, _ = self.certificate_manager.get_or_create(combiner.name).get_keypair_raw()
                import base64
                cert_b64 = base64.b64encode(combiner.certificate)
                response = {'host': combiner.address, 'port': combiner.port,
                            'certificate': str(cert_b64).split('\'')[1]}

                return jsonify(response)
            elif combiner is None:
                abort(404, description="Resource not found")
            # 1.receive client parameters
            # 2. check with available combiners if any clients are needed
            # 3. let client know where to connect.
            return

        @app.route('/infer')
        def infer():
            if self.control.state() == ReducerState.setup:
                return "Error, not configured"
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
        @app.route('/context', methods=['GET', 'POST'])
        @csrf.exempt #TODO fix csrf token to form posting in package.py
        def context():
            #if self.control.state() != ReducerState.setup or self.control.state() != ReducerState.idle:
            #    return "Error, Context already assigned!"
            if request.method == 'POST':

                if 'file' not in request.files:
                    flash('No file part')
                    return redirect(request.url)

                file = request.files['file']
                # if user does not select file, browser also
                # submit an empty part without filename
                if file.filename == '':
                    flash('No selected file')
                    return redirect(request.url)

                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                    if self.control.state() == ReducerState.instructing or self.control.state() == ReducerState.monitoring:
                        return "Not allowed to change context while execution is ongoing."
                    self.current_compute_context = filename #uploading new files will always set this to latest
                    self.control.set_compute_context(filename)
                    #return redirect(url_for('index',
                    #                        filename=filename))
                    return "success!"

            from flask import send_from_directory
            name = request.args.get('name', '')
            if name != '':
                return send_from_directory(app.config['UPLOAD_FOLDER'], name, as_attachment=True)
            if name == '' and self.current_compute_context:
                return send_from_directory(app.config['UPLOAD_FOLDER'], self.current_compute_context, as_attachment=True)

            return render_template('context.html')

        # import os, sys
        # self._original_stdout = sys.stdout
        # sys.stdout = open(os.devnull, 'w')
        if self.certificate:
            print("trying to connect with certs {} and key {}".format(str(self.certificate.cert_path),
                                                                      str(self.certificate.key_path)), flush=True)
            app.run(host="0.0.0.0", port="8090",
                    ssl_context=(str(self.certificate.cert_path), str(self.certificate.key_path)))
        # secure = False
        # secure_adhoc = False

        # if secure and secure_adhoc:
        #    app.run(host="0.0.0.0", port="8090", ssl_context='adhoc')
        # elif secure:
        #    app.run(host="0.0.0.0", port="8090", ssl_context=('cert.pem', 'key.pem'))
        # else:
        #    app.run(host="0.0.0.0", port="8090")
        # sys.stdout.close()
        # sys.stdout = self._original_stdout
