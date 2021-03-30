from fedn.clients.reducer.interfaces import CombinerInterface
from fedn.clients.reducer.state import ReducerState, ReducerStateToString
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename

from flask import Flask, jsonify, render_template, request
from flask import redirect, url_for, flash

from threading import Lock

import json
import plotly
import pandas as pd
import numpy
import math

import plotly.express as px
import geoip2.database
from fedn.clients.reducer.plots import Plot

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
        self.current_compute_context = None  # self.control.get_compute_context()

    def to_dict(self):
        data = {
            'name': self.name
        }
        return data

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

            client = self.name
            state = ReducerStateToString(self.control.state())
            logs = None
            refresh = True
            if self.current_compute_context == None or self.current_compute_context == '':
                return render_template('setup.html', client=client, state=state, logs=logs, refresh=False,
                                       message='Warning. No compute context is set. please set one with <a href="/context">/context</a>')

            if self.control.state() == ReducerState.setup:
                return render_template('setup.html', client=client, state=state, logs=logs, refresh=refresh,
                                       message='Warning. Reducer is not base-configured. please do so with config file.')

            return render_template('index.html', client=client, state=state, logs=logs, refresh=refresh)

        # http://localhost:8090/add?name=combiner&address=combiner&port=12080&token=e9a3cb4c5eaff546eec33ff68a7fbe232b68a192
        @app.route('/add')
        def add():

            """ Add a combiner to the network. """
            if self.control.state() == ReducerState.setup:
                return jsonify({'status': 'retry'})

            # TODO check for get variables
            name = request.args.get('name', None)
            address = str(request.args.get('address', None))
            port = request.args.get('port', None)
            # token = request.args.get('token')
            # TODO do validation

            if port is None or address is None or name is None:
                return "Please specify correct parameters."

            # Try to retrieve combiner from db
            combiner = self.control.network.get_combiner(name)
            if not combiner:
                # Create a new combiner
                import base64
                certificate, key = self.certificate_manager.get_or_create(address).get_keypair_raw()
                cert_b64 = base64.b64encode(certificate)
                key_b64 = base64.b64encode(key)

                # TODO append and redirect to index.
                import copy
                combiner = CombinerInterface(self, name, address, port, copy.deepcopy(certificate), copy.deepcopy(key),request.remote_addr)
                self.control.network.add_combiner(combiner)

            combiner = self.control.network.get_combiner(name)

            ret = {
                'status': 'added', 
                'certificate': combiner['certificate'],
                'key': combiner['key'], 
                'storage': self.control.statestore.get_storage_backend(),
                'statestore': self.control.statestore.get_config(),
            }                  

            return jsonify(ret)

        @app.route('/history', methods=['GET', 'POST'])
        def history():
            if request.method == 'POST':
                # upload seed file
                uploaded_seed = request.files['seed']
                if uploaded_seed:
                    from io import BytesIO
                    a = BytesIO()
                    a.seek(0, 0)
                    uploaded_seed.seek(0)
                    a.write(uploaded_seed.read())
                    helper = self.control.get_helper()                        
                    model = helper.load_model_from_BytesIO(a.getbuffer())
                    self.control.commit(uploaded_seed.filename, model)
            else:
                h_latest_model_id = self.control.get_latest_model()
                model_info = self.control.get_model_info()
                return render_template('index.html', h_latest_model_id=h_latest_model_id, seed=True,
                                       model_info=model_info)

            seed = True
            return redirect(url_for('history', seed=seed))

        @app.route('/delete_model_trail', methods=['GET', 'POST'])
        def delete_model_trail():
            if request.method == 'POST':
                from fedn.common.tracer.mongotracer import MongoTracer
                statestore_config = self.control.statestore.get_config()
                self.tracer = MongoTracer(statestore_config['mongo_config'], statestore_config['network_id'])
                try:
                    self.control.drop_models()
                except:
                    pass

                # drop objects in minio
                self.control.delete_bucket_objects()
                return redirect(url_for('history'))
            seed = True
            return redirect(url_for('history', seed=seed))

        @app.route('/drop_control', methods=['GET', 'POST'])
        def drop_control():
            if request.method == 'POST':
                self.control.statestore.drop_control()
                return redirect(url_for('start'))
            return redirect(url_for('start'))

        # http://localhost:8090/start?rounds=4&model_id=879fa112-c861-4cb1-a25d-775153e5b548
        @app.route('/start', methods=['GET', 'POST'])
        def start():
            client = self.name
            state = ReducerStateToString(self.control.state())
            logs = None
            refresh = True
            try:
                self.current_compute_context = self.control.get_compute_context()
            except:
                self.current_compute_context = None

            if self.current_compute_context == None or self.current_compute_context == '':
                return render_template('setup.html', client=client, state=state, logs=logs, refresh=False,
                                       message='No compute context is set. Please set one here <a href="/context">/context</a>')

            if self.control.state() == ReducerState.setup:
                return render_template('setup.html', client=client, state=state, logs=logs, refresh=refresh,
                                       message='Warning. Reducer is not base-configured. please do so with config file.')

            if request.method == 'POST':
                timeout = float(request.form.get('timeout'))
                rounds = int(request.form.get('rounds', 1))
                task = (request.form.get('task', ''))
                clients_required = request.form.get('clients_required', 1)
                clients_requested = request.form.get('clients_requested', 8)

                #TODO: Enable in UI
                validate = request.form.get('validate', False)
                helper_type = request.form.get('helper', 'keras')
                # self.control.statestore.set_framework(helper_type)

                latest_model_id = self.control.get_latest_model()

                config = {'round_timeout': timeout, 'model_id': latest_model_id,
                          'rounds': rounds, 'clients_required': clients_required,
                          'clients_requested': clients_requested, 'task': task,
                          'validate': validate, 'helper_type': helper_type}

                self.control.instruct(config)
                return redirect(url_for('index', message="Sent execution plan."))

            else:
                latest_model_id = self.control.get_latest_model()
                return render_template('index.html', latest_model_id=latest_model_id,
                                       compute_package=self.current_compute_context, helper=self.control.statestore.get_framework(), validate=True)

            client = self.name
            state = ReducerStateToString(self.control.state())
            logs = None
            refresh = False
            return render_template('index.html', client=client, state=state, logs=logs, refresh=refresh)

        @app.route('/assign')
        def assign():
            """Handle client assignment requests. """

            if self.control.state() == ReducerState.setup:
                return jsonify({'status': 'retry'})

            name = request.args.get('name', None)
            combiner_preferred = request.args.get('combiner', None)

            if combiner_preferred:
                combiner = self.control.find(combiner_preferred)
            else:
                combiner = self.control.find_available_combiner()

            client = {
                'name': name,
                'combiner_preferred': combiner_preferred,
                'ip': request.remote_addr,
                'status': 'available'
            }
            self.control.network.add_client(client)

            if combiner:
                import base64
                cert_b64 = base64.b64encode(combiner.certificate)
                response = {
                    'status': 'assigned',
                    'host': combiner.address,
                    'port': combiner.port,
                    'certificate': str(cert_b64).split('\'')[1],
                    'model_type': self.control.statestore.get_framework()
                }

                return jsonify(response)
            elif combiner is None:
                return jsonify({'status': 'retry'})

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

        def combiner_stats():
            combiner_info = []
            for combiner in self.control.network.get_combiners():
                try:
                    report = combiner.report()
                    combiner_info.append(report)
                except:
                    pass
                return combiner_info
            return False

        def create_map():
            cities_dict = {
                'city': [],
                'lat': [],
                'lon': [],
                'country': [],
                'name': [],
                'role': [],
                'size': []
            }

            from fedn import get_data
            dbpath = get_data('geolite2/GeoLite2-City.mmdb')

            with geoip2.database.Reader(dbpath) as reader:
                for combiner in self.control.statestore.list_combiners():
                    try:
                        response = reader.city(combiner['ip'])
                        cities_dict['city'].append(response.city.name)

                        r = 1.0  # Rougly 100km
                        w = r * math.sqrt(numpy.random.random())
                        t = 2.0 * math.pi * numpy.random.random()
                        x = w * math.cos(t)
                        y = w * math.sin(t)
                        lat = str(float(response.location.latitude) + x)
                        lon = str(float(response.location.longitude) + y)
                        cities_dict['lat'].append(lat)
                        cities_dict['lon'].append(lon)

                        cities_dict['country'].append(response.country.iso_code)

                        cities_dict['name'].append(combiner['name'])
                        cities_dict['role'].append('Combiner')
                        cities_dict['size'].append(10)

                    except geoip2.errors.AddressNotFoundError as err:
                        print(err)

            with geoip2.database.Reader(dbpath) as reader:
                for client in self.control.statestore.list_clients():
                    try:
                        response = reader.city(client['ip'])
                        cities_dict['city'].append(response.city.name)
                        cities_dict['lat'].append(response.location.latitude)
                        cities_dict['lon'].append(response.location.longitude)
                        cities_dict['country'].append(response.country.iso_code)

                        cities_dict['name'].append(client['name'])
                        cities_dict['role'].append('Client')
                        # TODO: Optionally relate to data size
                        cities_dict['size'].append(6)

                    except geoip2.errors.AddressNotFoundError as err:
                        print(err)

            config = self.control.statestore.get_config()

            cities_df = pd.DataFrame(cities_dict)
            if cities_df.empty:
                return False
            fig = px.scatter_geo(cities_df, lon="lon", lat="lat", projection="natural earth",
                                 color="role", size="size", hover_name="city",
                                 hover_data={"city": False, "lon": False, "lat": False, 'size': False,
                                             'name': True, 'role': True})

            fig.update_geos(fitbounds="locations", showcountries=True)
            fig.update_layout(title="FEDn network: {}".format(config['network_id']))

            fig = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return fig

        @app.route('/metric_type', methods=['GET', 'POST'])
        def change_features():
            feature = request.args['selected']
            plot = Plot(self.control.statestore)
            graphJSON = plot.create_box_plot(feature)
            return graphJSON

        @app.route('/dashboard')
        def dashboard():
            plot = Plot(self.control.statestore)
            try:
                valid_metrics = plot.fetch_valid_metrics()
                box_plot = plot.create_box_plot(valid_metrics[0])
            except:
                valid_metrics = None
                box_plot = None
            table_plot = plot.create_table_plot()
            # timeline_plot = plot.create_timeline_plot()
            timeline_plot = None
            clients_plot = plot.create_client_plot()
            return render_template('index.html', show_plot=True,
                                   box_plot=box_plot,
                                   table_plot=table_plot,
                                   timeline_plot=timeline_plot,
                                   clients_plot=clients_plot,
                                   metrics=valid_metrics
                                   )

        @app.route('/network')
        def network():
            plot = Plot(self.control.statestore)
            round_time_plot = plot.create_round_plot()
            mem_cpu_plot = plot.create_cpu_plot()
            combiners_plot = plot.create_combiner_plot()
            map_plot = create_map()
            combiner_info = combiner_stats()
            return render_template('index.html', map_plot=map_plot, network_plot=True,
                                   round_time_plot=round_time_plot,
                                   mem_cpu_plot=mem_cpu_plot,
                                   combiners_plot=combiners_plot,
                                   combiner_info=combiner_info
                                   )

        @app.route('/context', methods=['GET', 'POST'])
        @csrf.exempt  # TODO fix csrf token to form posting in package.py
        def context():
            # if self.control.state() != ReducerState.setup or self.control.state() != ReducerState.idle:
            #    return "Error, Context already assigned!"
            reset = request.args.get('reset', None)  # if reset is not empty then allow context re-set
            if reset:
                return render_template('context.html')

            if request.method == 'POST':

                if 'file' not in request.files:
                    flash('No file part')
                    return redirect(request.url)

                file = request.files['file']
                helper_type = request.form.get('helper', 'keras')
                # if user does not select file, browser also
                # submit an empty part without filename
                if file.filename == '':
                    flash('No selected file')
                    return redirect(request.url)

                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)

                    if self.control.state() == ReducerState.instructing or self.control.state() == ReducerState.monitoring:
                        return "Not allowed to change context while execution is ongoing."

                    self.control.set_compute_context(filename, file_path)
                    self.control.statestore.set_framework(helper_type)
                    return redirect(url_for('start'))

            from flask import send_from_directory
            name = request.args.get('name', '')

            if name == '':
                name = self.control.get_compute_context()
                if name == None or name == '':
                    return render_template('context.html')

            # There is a potential race condition here, if one client requests a package and at
            # the same time another one triggers a fetch from Minio and writes to disk. 
            try:
                mutex = Lock()
                mutex.acquire()
                return send_from_directory(app.config['UPLOAD_FOLDER'], name, as_attachment=True)
            except:
                try:
                    data = self.control.get_compute_package(name)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], name)
                    with open(file_path, 'wb') as fh:
                        fh.write(data)
                    return send_from_directory(app.config['UPLOAD_FOLDER'], name, as_attachment=True)
                except:
                    raise
            finally:
                mutex.release()

            return render_template('context.html')

        if self.certificate:
            print("trying to connect with certs {} and key {}".format(str(self.certificate.cert_path),
                                                                      str(self.certificate.key_path)), flush=True)
            app.run(host="0.0.0.0", port="8090",
                    ssl_context=(str(self.certificate.cert_path), str(self.certificate.key_path)))
