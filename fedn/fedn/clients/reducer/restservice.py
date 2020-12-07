from fedn.clients.reducer.interfaces import CombinerInterface
from fedn.clients.reducer.state import ReducerState, ReducerStateToString
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename

from flask import Flask, jsonify, render_template, request
from flask import redirect, url_for, flash

UPLOAD_FOLDER = '/app/client/package/'
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
        # TODO support CSRF in monitoring dashboard
        #dashboard.bind(app)
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
                return jsonify({'status': 'retry'})
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

        @app.route('/seed', methods=['GET', 'POST'])
        def seed():
            if request.method == 'POST':
                # upload seed file
                uploaded_seed = request.files['seed']
                if uploaded_seed:
                    self.control.commit(uploaded_seed.filename, uploaded_seed)
            else:
                h_latest_model_id = self.control.get_latest_model()
                model_info = self.control.get_model_info()
                return render_template('index.html', h_latest_model_id=h_latest_model_id, seed=True,
                                       model_info=model_info)

            seed = True
            return redirect(url_for('seed', seed=seed))

        @app.route('/delete', methods=['GET', 'POST'])
        def delete():
            if request.method == 'POST':
                from fedn.common.tracer.mongotracer import MongoTracer
                statestore_config = self.control.statestore.get_config()
                self.tracer = MongoTracer(statestore_config['mongo_config'], statestore_config['network_id'])
                try:
                    self.tracer.drop_round_time()
                    self.tracer.drop_ps_util_monitor()
                    self.tracer.drop_model_trail()
                    self.tracer.drop_latest_model()
                    self.tracer.drop_status()
                except:
                    pass

                # drop objects in minio
                self.control.delet_bucket_objects()
                return redirect(url_for('seed'))
            seed = True
            return redirect(url_for('seed', seed=seed))

        @app.route('/drop_db', methods=['GET', 'POST'])
        def drop_db():
            if request.method == 'POST':
                from fedn.common.storage.db.mongo import drop_mongodb
                statestore_config = self.control.statestore.get_config()
                self.mdb = drop_mongodb(statestore_config['mongo_config'], statestore_config['network_id'])
                return redirect(url_for('seed'))
            seed = True
            return redirect(url_for('seed', seed=seed))

        # http://localhost:8090/start?rounds=4&model_id=879fa112-c861-4cb1-a25d-775153e5b548
        @app.route('/start', methods=['GET', 'POST'])
        def start():
            if self.control.state() == ReducerState.setup:
                return "Error, not configured"

            if request.method == 'POST':
                timeout = request.form.get('timeout', 180)
                rounds = int(request.form.get('rounds', 1))

                task = (request.form.get('task', ''))
                active_clients = request.form.get('active_clients', 2)
                clients_required = request.form.get('clients_required', 1)
                clients_requested = request.form.get('clients_requested', 8)

                latest_model_id = self.control.get_latest_model()
                config = {'round_timeout': timeout, 'model_id': latest_model_id,
                          'rounds': rounds, 'active_clients': active_clients, 'clients_required': clients_required,
                          'clients_requested': clients_requested, 'task': task}

                # from fedn.common.tracer.mongotracer import MongoTracer
                # self.tracer = MongoTracer()
                # self.tracer.ps_util_monitor(target=self.control.instruct(config))
                self.control.instruct(config)
                return redirect(url_for('index', message="Sent execution plan."))

            else:
                # Select rounds UI
                rounds = range(1, 500)
                latest_model_id = self.control.get_latest_model()
                return render_template('index.html', round_options=rounds, latest_model_id=latest_model_id)

            client = self.name
            state = ReducerStateToString(self.control.state())
            logs = None
            refresh = False
            return render_template('index.html', client=client, state=state, logs=logs, refresh=refresh)

        @app.route('/assign')
        def assign():
            if self.control.state() == ReducerState.setup:
                return jsonify({'status': 'retry'})
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
                response = {'status': 'assigned', 'host': combiner.address, 'port': combiner.port,
                            'certificate': str(cert_b64).split('\'')[1]}

                return jsonify(response)
            elif combiner is None:
                return jsonify({'status':'retry'})

            return jsonify({'status': 'retry'})

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


        @app.route('/plot')
        def plot():
            box = 'box'
            plot = create_plot(box)
            show_plot = True
            return render_template('index.html', show_plot=show_plot, plot=plot)

        def create_plot(feature):
            from fedn.clients.reducer.plots import Plot

            plot = Plot(self.control.statestore)

            if feature == 'table':
                return plot.create_table_plot()
            elif feature == 'timeline':
                return plot.create_timeline_plot()
            elif feature == 'round_time':
                return plot.create_round_plot()
            elif feature == 'box':
                return plot.create_box_plot()
            elif feature == 'cpu':
                return plot.create_cpu_plot()
            else:
                return 'No plot!'

        @app.route('/plot_type', methods=['GET', 'POST'])
        def change_features():
            feature = request.args['selected']
            graphJSON = create_plot(feature)
            return graphJSON


        @app.route('/context', methods=['GET', 'POST'])
        @csrf.exempt  # TODO fix csrf token to form posting in package.py
        def context():
            # if self.control.state() != ReducerState.setup or self.control.state() != ReducerState.idle:
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
                    self.current_compute_context = filename  # uploading new files will always set this to latest
                    self.control.set_compute_context(filename)
                    # return redirect(url_for('index',
                    #                        filename=filename))
                    return "success!"

            from flask import send_from_directory
            name = request.args.get('name', '')
            if name != '':
                return send_from_directory(app.config['UPLOAD_FOLDER'], name, as_attachment=True)
            if name == '' and self.current_compute_context:
                return send_from_directory(app.config['UPLOAD_FOLDER'], self.current_compute_context,
                                           as_attachment=True)

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
