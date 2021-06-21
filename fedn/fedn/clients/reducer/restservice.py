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
ALLOWED_EXTENSIONS = {'gz', 'bz2', 'tar', 'zip', 'tgz'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class ReducerRestService:
    def __init__(self, config, control, certificate_manager, certificate=None):

        print("config object!: \n\n\n\n{}".format(config))
        if config['discover_host']:
            self.name = config['discover_host']
        else:
            self.name = config['name']
        self.port = config['discover_port']
        self.network_id = config['name'] + '-network'

        if not config['token']:
            import uuid
            self.token = str(uuid.uuid4())
        else:
            self.token = config['token']

        self.control = control
        self.certificate = certificate
        self.certificate_manager = certificate_manager
        self.current_compute_context = None  # self.control.get_compute_context()

    def to_dict(self):
        data = {
            'name': self.name
        }
        return data

    def check_configured(self):
        if not self.control.get_compute_context():
            return render_template('setup.html', client=self.name, state=ReducerStateToString(self.control.state()),
                                   logs=None, refresh=False,
                                   message='')

        if self.control.state() == ReducerState.setup:
            return render_template('setup.html', client=self.name, state=ReducerStateToString(self.control.state()),
                                   logs=None, refresh=True,
                                   message='Warning. Reducer is not base-configured. please do so with config file.')

        if not self.control.get_latest_model():
            return render_template('setup_model.html', message="Please set the initial model.")

        return None

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
            not_configured = self.check_configured()
            if not_configured:
                return not_configured
            events = self.control.get_events()
            message = request.args.get('message', None)
            message_type = request.args.get('message_type', None)
            return render_template('events.html', client=self.name, state=ReducerStateToString(self.control.state()),
                                   events=events,
                                   logs=None, refresh=True, configured=True, message=message, message_type=message_type)

        # http://localhost:8090/add?name=combiner&address=combiner&port=12080&token=e9a3cb4c5eaff546eec33ff68a7fbe232b68a192
        @app.route('/status')
        def status():
            return {'state': ReducerStateToString(self.control.state())}

        @app.route('/netgraph')
        def netgraph():

            result = {'nodes': [], 'edges': []}

            result['nodes'].append({
                "id": "r0",
                "label": "Reducer",
                "x": -1.2,
                "y": 0,
                "size": 25,
                "type": 'reducer',
            })
            x = 0
            y = 0
            count = 0
            meta = {}
            combiner_info = []
            for combiner in self.control.network.get_combiners():
                try:
                    report = combiner.report()
                    combiner_info.append(report)
                except:
                    pass
            y = y + 0.5
            width = 5
            if len(combiner_info) < 1:
                return result
            step = 5 / len(combiner_info)
            x = -width / 3.0
            for combiner in combiner_info:
                print("combiner info {}".format(combiner_info), flush=True)

                try:
                    result['nodes'].append({
                        "id": combiner['name'],  # "n{}".format(count),
                        "label": "Combiner ({} clients)".format(combiner['nr_active_clients']),
                        "x": x,
                        "y": y,
                        "size": 15,
                        "name": combiner['name'],
                        "type": 'combiner',
                        # "color":'blue',
                    })
                except Exception as err:
                    print(err)

                x = x + step
                count = count + 1
            y = y + 0.25

            count = 0
            width = 5
            step = 5 / len(combiner_info)
            x = -width / 2.0
            # for combiner in self.control.statestore.list_clients():
            for combiner in combiner_info:
                for a in range(0, int(combiner['nr_active_clients'])):
                    # y = y + 0.25
                    try:
                        result['nodes'].append({
                            "id": "c{}".format(count),
                            "label": "Client",
                            "x": x,
                            "y": y,
                            "size": 15,
                            "name": "c{}".format(count),
                            "combiner": combiner['name'],
                            "type": 'client',
                            # "color":'blue',
                        })
                    except Exception as err:
                        print(err)
                    # print("combiner prefferred name {}".format(client['combiner']), flush=True)
                    x = x + 0.25
                    count = count + 1

            count = 0
            for node in result['nodes']:
                try:
                    if node['type'] == 'combiner':
                        result['edges'].append(
                            {
                                "id": "e{}".format(count),
                                "source": node['id'],
                                "target": 'r0',
                            }
                        )
                    elif node['type'] == 'client':
                        result['edges'].append(
                            {
                                "id": "e{}".format(count),
                                "source": node['combiner'],
                                "target": node['id'],
                            }
                        )
                except Exception as e:
                    pass
                count = count + 1

            return result

        @app.route('/events')
        def events():
            import json
            from bson import json_util

            json_docs = []
            for doc in self.control.get_events():
                json_doc = json.dumps(doc, default=json_util.default)
                json_docs.append(json_doc)

            json_docs.reverse()
            return {'events': json_docs}

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
                combiner = CombinerInterface(self, name, address, port, copy.deepcopy(certificate), copy.deepcopy(key),
                                             request.remote_addr)
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

        @app.route('/eula', methods=['GET', 'POST'])
        def eula():
            for r in request.headers:
                print("header contains: {}".format(r), flush=True)

            return render_template('eula.html', configured=True)

        @app.route('/models', methods=['GET', 'POST'])
        def models():

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
                not_configured = self.check_configured()
                if not_configured:
                    return not_configured
                h_latest_model_id = self.control.get_latest_model()

                model_info = self.control.get_model_info()
                return render_template('models.html', h_latest_model_id=h_latest_model_id, seed=True,
                                       model_info=model_info, configured=True)

            seed = True
            return redirect(url_for('models', seed=seed))

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
                return redirect(url_for('models'))
            seed = True
            return redirect(url_for('models', seed=seed))

        @app.route('/drop_control', methods=['GET', 'POST'])
        def drop_control():
            if request.method == 'POST':
                self.control.statestore.drop_control()
                return redirect(url_for('control'))
            return redirect(url_for('control'))

        # http://localhost:8090/control?rounds=4&model_id=879fa112-c861-4cb1-a25d-775153e5b548
        @app.route('/control', methods=['GET', 'POST'])
        def control():
            not_configured = self.check_configured()
            if not_configured:
                return not_configured
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

            if self.control.state() == ReducerState.monitoring:
                return redirect(
                    url_for('index', state=state, refresh=refresh, message="Reducer is in monitoring state"))

            if request.method == 'POST':
                timeout = float(request.form.get('timeout', 180))
                rounds = int(request.form.get('rounds', 1))
                task = (request.form.get('task', ''))
                clients_required = request.form.get('clients_required', 1)
                clients_requested = request.form.get('clients_requested', 8)

                # checking if there are enough clients connected to start!
                clients_available = 0
                try:
                    for combiner in self.control.network.get_combiners():
                        if combiner.allowing_clients():
                            combiner_state = combiner.report()
                            nac = combiner_state['nr_active_clients']

                            clients_available = clients_available + int(nac)
                except Exception as e:
                    pass

                if clients_available < clients_required:
                    return redirect(url_for('index', state=state,
                                            message="Not enough clients available to start rounds.",
                                            message_type='warning'))

                validate = request.form.get('validate', False)
                if validate == 'False':
                    validate = False
                helper_type = request.form.get('helper', 'keras')
                # self.control.statestore.set_framework(helper_type)

                latest_model_id = self.control.get_latest_model()

                config = {'round_timeout': timeout, 'model_id': latest_model_id,
                          'rounds': rounds, 'clients_required': clients_required,
                          'clients_requested': clients_requested, 'task': task,
                          'validate': validate, 'helper_type': helper_type}

                import threading
                threading.Thread(target=self.control.instruct, args=(config,)).start()
                # self.control.instruct(config)
                return redirect(url_for('index', state=state, refresh=refresh, message="Sent execution plan.",
                                        message_type='SUCCESS'))

            else:
                seed_model_id = None
                latest_model_id = None
                try:
                    seed_model_id = self.control.get_first_model()[0]
                    latest_model_id = self.control.get_latest_model()
                except Exception as e:
                    pass

                return render_template('index.html', latest_model_id=latest_model_id,
                                       compute_package=self.current_compute_context,
                                       seed_model_id=seed_model_id,
                                       helper=self.control.statestore.get_framework(), validate=True, configured=True)

            client = self.name
            state = ReducerStateToString(self.control.state())
            logs = None
            refresh = False
            return render_template('index.html', client=client, state=state, logs=logs, refresh=refresh,
                                   configured=True)

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

            if combiner is None:
                return jsonify({'status': 'retry'})
            ## Check that a framework has been selected prior to assigning clients.
            framework = self.control.statestore.get_framework()
            if not framework:
                return jsonify({'status': 'retry'})

            client = {
                'name': name,
                'combiner_preferred': combiner_preferred,
                'combiner': combiner.name,
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
                    'ip': combiner.ip,
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
            not_configured = self.check_configured()
            if not_configured:
                return not_configured

            plot = Plot(self.control.statestore)
            try:
                valid_metrics = plot.fetch_valid_metrics()
                box_plot = plot.create_box_plot(valid_metrics[0])
            except Exception as e:
                valid_metrics = None
                box_plot = None
                print(e, flush=True)
            table_plot = plot.create_table_plot()
            # timeline_plot = plot.create_timeline_plot()
            timeline_plot = None
            clients_plot = plot.create_client_plot()
            return render_template('dashboard.html', show_plot=True,
                                   box_plot=box_plot,
                                   table_plot=table_plot,
                                   timeline_plot=timeline_plot,
                                   clients_plot=clients_plot,
                                   metrics=valid_metrics,
                                   configured=True
                                   )

        @app.route('/network')
        def network():
            not_configured = self.check_configured()
            if not_configured:
                return not_configured
            plot = Plot(self.control.statestore)
            round_time_plot = plot.create_round_plot()
            mem_cpu_plot = plot.create_cpu_plot()
            combiners_plot = plot.create_combiner_plot()
            map_plot = create_map()
            combiner_info = combiner_stats()
            return render_template('network.html', map_plot=map_plot, network_plot=True,
                                   round_time_plot=round_time_plot,
                                   mem_cpu_plot=mem_cpu_plot,
                                   combiners_plot=combiners_plot,
                                   combiner_info=combiner_info,
                                   configured=True
                                   )

        @app.route('/config/download', methods=['GET'])
        def config_download():

            chk_string = ""
            name = self.control.get_compute_context()
            if name is None or name == '':
                chk_string = ''
            else:
                file_path = os.path.join(UPLOAD_FOLDER, name)
                print("trying to get {}".format(file_path))
                from fedn.utils.checksum import md5

                try:
                    sum = str(md5(file_path))
                except FileNotFoundError as e:
                    sum = ''
                chk_string = "checksum: {}".format(sum)

            network_id = self.network_id
            discover_host = self.name
            discover_port = self.port
            token = self.token
            ctx = """network_id: {network_id}
controller:
    discover_host: {discover_host}
    discover_port: {discover_port}
    token: {token}
    {chk_string}""".format(network_id=network_id,
                             discover_host=discover_host,
                             discover_port=discover_port,
                             token=token,
                             chk_string=chk_string)

            from io import BytesIO
            from flask import send_file
            obj = BytesIO()
            obj.write(ctx.encode('UTF-8'))
            obj.seek(0)
            return send_file(obj,
                             as_attachment=True,
                             attachment_filename='client.yaml',
                             mimetype='application/x-yaml')


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
                    return redirect(url_for('context'))

                file = request.files['file']
                helper_type = request.form.get('helper', 'keras')
                # if user does not select file, browser also
                # submit an empty part without filename
                if file.filename == '':
                    flash('No selected file')
                    return redirect(url_for('context'))

                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)

                    if self.control.state() == ReducerState.instructing or self.control.state() == ReducerState.monitoring:
                        return "Not allowed to change context while execution is ongoing."

                    self.control.set_compute_context(filename, file_path)
                    self.control.statestore.set_framework(helper_type)
                    return redirect(url_for('control'))

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

        @app.route('/checksum', methods=['GET', 'POST'])
        def checksum():

            #sum = ''
            name = request.args.get('name', None)
            if name == '' or name is None:
                name = self.control.get_compute_context()
                if name == None or name == '':
                    return jsonify({})

            file_path = os.path.join(UPLOAD_FOLDER, name)
            print("trying to get {}".format(file_path))
            from fedn.utils.checksum import md5

            try:
                sum = str(md5(file_path))
            except FileNotFoundError as e:
                sum = ''

            data = {'checksum': sum}
            from flask import jsonify
            return jsonify(data)

        if self.certificate:
            print("trying to connect with certs {} and key {}".format(str(self.certificate.cert_path),
                                                                      str(self.certificate.key_path)), flush=True)
            app.run(host="0.0.0.0", port=self.port,
                    ssl_context=(str(self.certificate.cert_path), str(self.certificate.key_path)))
