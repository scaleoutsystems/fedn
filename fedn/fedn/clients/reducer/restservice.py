from fedn.clients.reducer.interfaces import CombinerInterface
from fedn.clients.reducer.state import ReducerState, ReducerStateToString
from flask import Flask, flash
from flask import jsonify, abort
from flask import render_template
from flask import request, redirect, url_for
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/app/client/package/'
ALLOWED_EXTENSIONS = {'gz', 'bz2', 'tar', 'zip'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


from flask import Flask, jsonify, render_template, request
from flask import redirect, url_for
import pymongo
import json
import numpy
import plotly.graph_objs as go
from datetime import datetime,timedelta
import plotly

import os
import flask_monitoringdashboard as dashboard


class ReducerRestService:
    def __init__(self, name, control, certificate_manager, certificate=None):
        self.name = name
        self.control = control
        self.certificate = certificate
        self.certificate_manager = certificate_manager

        self.current_compute_context = self.control.get_compute_context()

    def run(self):
        app = Flask(__name__)
        #dashboard.bind(app)
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        csrf = CSRFProtect()
        import os
        SECRET_KEY = os.urandom(32)
        app.config['SECRET_KEY'] = SECRET_KEY

        mc = pymongo.MongoClient(os.environ['FEDN_MONGO_HOST'], int(os.environ['FEDN_MONGO_PORT']), username=os.environ['FEDN_MONGO_USER'],
                                 password=os.environ['FEDN_MONGO_PASSWORD'])
        mdb = mc[os.environ['ALLIANCE_UID']]
        alliance = mdb["status"]
        round_time = mdb["performances"]

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
                clients_required = request.form.get('clients_required', 2)
                clients_requested = request.form.get('clients_requested', 8)

                latest_model_id = self.control.get_latest_model()
                config = {'round_timeout': timeout, 'model_id': latest_model_id,
                          'rounds': rounds, 'active_clients': active_clients, 'clients_required': clients_required,
                          'clients_requested': clients_requested, 'task': task}

                self.control.instruct(config)
                return redirect(url_for('index', message="Sent execution plan."))

            else:
                # Select rounds UI
                rounds = range(1, 100)
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
                #abort(404, description="Resource not found")
            # 1.receive client parameters
            # 2. check with available combiners if any clients are needed
            # 3. let client know where to connect.
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

        # plot metrics from DB
        def _scalar_metrics(metrics):
            """ Extract all scalar valued metrics from a MODEL_VALIDATON. """

            data = json.loads(metrics['data'])
            data = json.loads(data['data'])

            valid_metrics = []
            for metric, val in data.items():
                # If it can be converted to a float it is a valid, scalar metric
                try:
                    val = float(val)
                    valid_metrics.append(metric)
                except:
                    pass

            return valid_metrics

        @app.route('/plot')
        def plot():
            box = 'box'
            plot = create_plot(box)
            show_plot = True
            return render_template('index.html', show_plot=show_plot, plot=plot)

        def create_plot(feature):
            if feature == 'table':
                return create_table_plot()
            elif feature == 'timeline':
                return create_timeline_plot()
            elif feature == 'round_time':
                return create_round_plot()
            elif feature == 'box':
                return create_box_plot()
            else:
                return 'No plot!'

        @app.route('/plot_type', methods=['GET', 'POST'])
        def change_features():
            feature = request.args['selected']
            graphJSON = create_plot(feature)
            return graphJSON

        def create_table_plot():
            metrics = alliance.find_one({'type': 'MODEL_VALIDATION'})
            if metrics == None:
                fig = go.Figure(data=[])
                fig.update_layout(title_text='No data currently available for mean metrics')
                table = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                return table

            valid_metrics = _scalar_metrics(metrics)
            if valid_metrics == []:
                fig = go.Figure(data=[])
                fig.update_layout(title_text='No scalar metrics found')
                table = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                return table

            all_vals = []
            models = []
            for metric in valid_metrics:
                validations = {}
                for post in alliance.find({'type': 'MODEL_VALIDATION'}):
                    e = json.loads(post['data'])
                    try:
                        validations[e['modelId']].append(float(json.loads(e['data'])[metric]))
                    except KeyError:
                        validations[e['modelId']] = [float(json.loads(e['data'])[metric])]

                vals = []
                models = []
                for model, data in validations.items():
                    vals.append(numpy.mean(data))
                    models.append(model)
                all_vals.append(vals)

            header_vals = valid_metrics
            models.reverse()
            values = [models]
            print(all_vals, flush=True)
            for vals in all_vals:
                vals.reverse()
                values.append(vals)

            fig = go.Figure(data=[go.Table(
                header=dict(values=['Model ID'] + header_vals,
                            line_color='darkslategray',
                            fill_color='lightskyblue',
                            align='left'),

                cells=dict(values=values,  # 2nd column
                           line_color='darkslategray',
                           fill_color='lightcyan',
                           align='left'))
            ])

            fig.update_layout(title_text='Summary: mean metrics')
            table = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return table

        def create_timeline_plot():
            trace_data = []
            x = []
            y = []
            base = []
            for p in alliance.find({'type': 'MODEL_UPDATE_REQUEST'}):
                e = json.loads(p['data'])
                cid = e['correlationId']
                for cc in alliance.find({'sender': p['sender'], 'type': 'MODEL_UPDATE'}):
                    da = json.loads(cc['data'])
                    if da['correlationId'] == cid:
                        cp = cc

                cd = json.loads(cp['data'])
                tr = datetime.strptime(e['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
                tu = datetime.strptime(cd['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
                ts = tu - tr
                base.append(tr.timestamp())
                x.append(ts.total_seconds())
                y.append(p['sender']['name'])

            trace_data.append(go.Bar(
                x=x,
                y=y,
                orientation='h',
                base=base,
                marker=dict(color='royalblue'),
                name="Training",
            ))

            x = []
            y = []
            base = []
            for p in alliance.find({'type': 'MODEL_VALIDATION_REQUEST'}):
                e = json.loads(p['data'])
                cid = e['correlationId']
                for cc in alliance.find({'sender': p['sender'], 'type': 'MODEL_VALIDATION'}):
                    da = json.loads(cc['data'])
                    if da['correlationId'] == cid:
                        cp = cc
                cd = json.loads(cp['data'])
                tr = datetime.strptime(e['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
                tu = datetime.strptime(cd['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
                ts = tu - tr
                base.append(tr.timestamp())
                x.append(ts.total_seconds())
                y.append(p['sender']['name'])

            trace_data.append(go.Bar(
                x=x,
                y=y,
                orientation='h',
                base=base,
                marker=dict(color='lightskyblue'),
                name="Validation",
            ))

            layout = go.Layout(
                barmode='stack',
                showlegend=True,
            )

            fig = go.Figure(data=trace_data, layout=layout)
            fig.update_xaxes(title_text='Timestamp')
            fig.update_layout(title_text='Alliance timeline')
            timeline = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return timeline

        def create_ml_plot():
            metrics = alliance.find_one({'type': 'MODEL_VALIDATION'})
            if metrics == None:
                fig = go.Figure(data=[])
                fig.update_layout(title_text='No data currently available for Mean Absolute Error')
                ml = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                return ml

            data = json.loads(metrics['data'])
            data = json.loads(data['data'])
            valid_metrics = []
            for metric, val in data.items():
                # Check if scalar - is this robust ?
                if isinstance(val, float):
                    valid_metrics.append(metric)

            # Assemble a dict with all validations
            validations = {}
            clients = {}

            for post in alliance.find({'type': 'MODEL_VALIDATION'}):
                try:
                    e = json.loads(post['data'])
                    clients[post['sender']['name']].append(json.loads(e['data'])[metric])
                except KeyError:
                    clients[post['sender']['name']] = []

            rounds = []
            traces_data = []

            for c in clients:
                traces_data.append(go.Scatter(
                    x=rounds,
                    y=clients[c],
                    name=c
                ))
            fig = go.Figure(traces_data)
            fig.update_xaxes(title_text='Rounds')
            fig.update_yaxes(title_text='MAE', tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            fig.update_layout(title_text='Mean Absolute Error Plot')
            ml = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return ml

        def create_box_plot():
            metrics = alliance.find_one({'type': 'MODEL_VALIDATION'})
            if metrics == None:
                fig = go.Figure(data=[])
                fig.update_layout(title_text='No data currently available for metric distribution over alliance '
                                             'participants')
                box = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                return box

            valid_metrics = _scalar_metrics(metrics)
            if valid_metrics == []:
                fig = go.Figure(data=[])
                fig.update_layout(title_text='No scalar metrics found')
                box = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                return box

            # Just grab the first metric in the list.
            # TODO: Let the user choose, or plot all of them.
            if "accuracy" in valid_metrics:
                metric = "accuracy"
            else:
                metric = valid_metrics[0]
            validations = {}
            for post in alliance.find({'type': 'MODEL_VALIDATION'}):
                e = json.loads(post['data'])
                try:
                    validations[e['modelId']].append(float(json.loads(e['data'])[metric]))
                except KeyError:
                    validations[e['modelId']] = [float(json.loads(e['data'])[metric])]

            box = go.Figure()

            x = []
            y = []
            box_trace = []
            for model_id, acc in validations.items():
                x.append(model_id)
                y.append(numpy.mean([float(i) for i in acc]))
                if len(acc) >= 2:
                    box.add_trace(go.Box(y=acc, name=str(model_id), marker_color="royalblue", showlegend=False))

            rounds = list(range(len(y)))
            box.add_trace(go.Scatter(
                x=x,
                y=y,
                name='Mean'
            ))

            box.update_xaxes(title_text='Model ID')
            box.update_yaxes(tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            box.update_layout(title_text='Metric distribution over alliance participants: {}'.format(metric))
            box = json.dumps(box, cls=plotly.utils.PlotlyJSONEncoder)
            return box

        def create_round_plot():
            metrics = round_time.find_one({'key': 'performance'})
            if metrics == None:
                fig = go.Figure(data=[])
                fig.update_layout(title_text='No data currently available for round time')
                ml = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                return ml

            for post in round_time.find({'key': 'performance'}):
                rounds = post['round']
                traces_data = post['time']

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rounds,
                y=traces_data,
                mode='lines+markers',
                name='Time'
            ))

            fig.update_xaxes(title_text='Rounds')
            fig.update_yaxes(title_text='Time (s)')
            fig.update_layout(title_text='Round time')
            round_t = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return round_t

        # @app.route('/seed')
        # def seed():
        #    try:
        #        result = self.inference.infer(request.args)
        #    except fedn.exceptions.ModelError:
        #        print("no model")
        #
        #    return result
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
