import base64
import copy
import datetime
import json
import os
import re
import threading
from io import BytesIO
from threading import Lock

import jwt
import pandas as pd
from bokeh.embed import json_item
from bson import json_util
from flask import (Flask, abort, flash, jsonify, make_response, redirect,
                   render_template, request, send_file, send_from_directory,
                   url_for)
from flask_talisman import Talisman
from werkzeug.utils import secure_filename

from fedn.clients.reducer.interfaces import CombinerInterface
from fedn.clients.reducer.plots import Plot
from fedn.clients.reducer.state import ReducerState, ReducerStateToString
from fedn.common.exceptions import ModelError
from fedn.common.tracer.mongotracer import MongoTracer
from fedn.utils.checksum import sha

UPLOAD_FOLDER = '/app/client/package/'
ALLOWED_EXTENSIONS = {'gz', 'bz2', 'tar', 'zip', 'tgz'}


def allowed_file(filename):
    """

    :param filename:
    :return:
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_auth_token(secret_key):
    """Generates the Auth Token
    :return: string
    """
    try:
        payload = {
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=90, seconds=0),
            'iat': datetime.datetime.utcnow(),
            'status': 'Success'
        }
        token = jwt.encode(
            payload,
            secret_key,
            algorithm='HS256'
        )
        print('\n\n\nSECURE MODE ENABLED, USE TOKEN TO ACCESS REDUCER: **** {} ****\n\n\n'.format(token))
        return token
    except Exception as e:
        return e


def decode_auth_token(auth_token, secret):
    """Decodes the auth token
    :param auth_token:
    :return: string
    """
    try:
        payload = jwt.decode(
            auth_token,
            secret,
            algorithms=['HS256']
        )
        return payload["status"]
    except jwt.ExpiredSignatureError as e:
        print(e)
        return 'Token has expired.'
    except jwt.InvalidTokenError as e:
        print(e)
        return 'Invalid token.'


class ReducerRestService:
    """

    """

    def __init__(self, config, control, certificate_manager, certificate=None):

        print("config object!: \n\n\n\n{}".format(config))
        if config['discover_host']:
            self.name = config['discover_host']
        else:
            self.name = config['name']

        self.port = config['discover_port']
        self.network_id = config['name'] + '-network'

        if 'token' in config.keys():
            self.token_auth_enabled = True
        else:
            self.token_auth_enabled = False

        if 'secret_key' in config.keys():
            self.SECRET_KEY = config['secret_key']
        else:
            self.SECRET_KEY = None

        if 'secure' in config.keys():
            self.secure = config['secure']

        self.remote_compute_context = config["remote_compute_context"]
        if self.remote_compute_context:
            self.package = 'remote'
        else:
            self.package = 'local'

        self.control = control
        self.certificate = certificate
        self.certificate_manager = certificate_manager
        self.current_compute_context = None  # self.control.get_compute_context()

    def to_dict(self):
        """

        :return:
        """
        data = {
            'name': self.name
        }
        return data

    def check_compute_context(self):
        """Check if the compute context/package has been configured,
        if remote compute context is set to False, True will be returned

        :return: True if configured
        :rtype: bool
        """
        if not self.remote_compute_context:
            return True

        if not self.control.get_compute_context():
            return False
        else:
            return True

    def check_initial_model(self):
        """Check if initial model (seed model) has been configured

        :return: True if configured, else False
        :rtype: bool
        """

        if self.control.get_latest_model():
            return True
        else:
            return False

    def check_configured_response(self):
        """Check if everything has been configured for client to connect,
        return response if not.

        :return: Reponse with message if not configured, else None
        :rtype: json
        """
        if self.control.state() == ReducerState.setup:
            return jsonify({'status': 'retry',
                            'package': self.package,
                            'msg': "Controller is not configured."})

        if not self.check_compute_context():
            return jsonify({'status': 'retry',
                            'package': self.package,
                            'msg': "Compute package is not configured. Please upload the compute package."})

        if not self.check_initial_model():
            return jsonify({'status': 'retry',
                            'package': self.package,
                            'msg': "Initial model is not configured. Please upload the model."})

        if not self.control.idle():
            return jsonify({'status': 'retry',
                            'package': self.package,
                            'msg': "Conroller is not in idle state, try again later. "})
        return None

    def check_configured(self):
        """Check if compute package has been configured and that and that the
        state of the ReducerControl is not in setup otherwise render setup template.
        Check if initial model has been configured, otherwise render setup_model template.
        :return: Rendered html template or None
        """
        if not self.check_compute_context():
            return render_template('setup.html', client=self.name, state=ReducerStateToString(self.control.state()),
                                   logs=None, refresh=False,
                                   message='Please set the compute package')

        if self.control.state() == ReducerState.setup:
            return render_template('setup.html', client=self.name, state=ReducerStateToString(self.control.state()),
                                   logs=None, refresh=True,
                                   message='Warning. Reducer is not base-configured. please do so with config file.')

        if not self.check_initial_model():
            return render_template('setup_model.html', message="Please set the initial model.")

        return None

    def authorize(self, r, secret):
        """Authorize client token

        :param r: Request
        :type r: [type]
        :param token: Token to verify against
        :type token: string
        """
        try:
            # Get token
            if 'Authorization' in r.headers:  # header auth
                request_token = r.headers.get('Authorization').split()[1]
            elif 'token' in r.args:  # args auth
                request_token = str(r.args.get('token'))
            elif 'fedn_token' in r.cookies:
                request_token = r.cookies.get('fedn_token')
            else:  # no token provided
                print('Authorization failed. No token provided.', flush=True)
                abort(401)

            # Log token and secret
            print(
                f'Secret: {secret}. Request token: {request_token}.', flush=True)

            # Authenticate
            status = decode_auth_token(request_token, secret)
            if status == 'Success':
                return True
            else:
                print('Authorization failed. Status: "{}"'.format(
                    status), flush=True)
                abort(401)
        except Exception as e:
            print('Authorization failed. Expection encountered: "{}".'.format(
                e), flush=True)
            abort(401)

    def run(self):
        """

        :return:
        """
        app = Flask(__name__)
        if self.secure:
            # Among other things, force https
            Talisman(app, content_security_policy=None)

        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        app.config['SECRET_KEY'] = self.SECRET_KEY

        # @app.before_request
        # def before_request_callback():
        #    """ If configured with SSL, forward http to https. """
        #    #if not request.is_secure:
        #    if self.secure:
        #        url = request.url.replace("http://", "https://", 1)
        #        code = 301
        #        print(url,flush=True)
        #        return redirect(url, code=code)

        @app.route('/')
        def index():
            """

            :return:
            """
            # Token auth
            if self.token_auth_enabled:
                self.authorize(request, app.config.get('SECRET_KEY'))

            # Render template
            not_configured_template = self.check_configured()
            if not_configured_template:
                template = not_configured_template
            else:
                events = self.control.get_events()
                message = request.args.get('message', None)
                message_type = request.args.get('message_type', None)
                template = render_template('events.html', client=self.name, state=ReducerStateToString(self.control.state()),
                                           events=events,
                                           logs=None, refresh=True, configured=True, message=message, message_type=message_type)

            # Set token cookie in response if needed
            response = make_response(template)
            if 'token' in request.args:  # args auth
                response.set_cookie('fedn_token', str(request.args['token']))

            # Return response
            return response

        # http://localhost:8090/add?name=combiner&address=combiner&port=12080&token=e9a3cb4c5eaff546eec33ff68a7fbe232b68a192
        @app.route('/status')
        def status():
            """

            :return:
            """
            return {'state': ReducerStateToString(self.control.state())}

        @app.route('/netgraph')
        def netgraph():
            """
            Creates nodes and edges for network graph

            :return: nodes and edges as keys
            :rtype: dict
            """
            result = {'nodes': [], 'edges': []}

            result['nodes'].append({
                "id": "reducer",
                "label": "Reducer",
                "role": 'reducer',
                "status": 'active',
                "name": 'reducer',  # TODO: get real host name
                "type": 'reducer',
            })

            combiner_info = combiner_status()
            client_info = client_status()

            if len(combiner_info) < 1:
                return result

            for combiner in combiner_info:
                print("combiner info {}".format(combiner_info), flush=True)
                try:
                    result['nodes'].append({
                        "id": combiner['name'],  # "n{}".format(count),
                        "label": "Combiner ({} clients)".format(combiner['nr_active_clients']),
                        "role": 'combiner',
                        "status": 'active',  # TODO: Hard-coded, combiner_info does not contain status
                        "name": combiner['name'],
                        "type": 'combiner',
                    })
                except Exception as err:
                    print(err)

            for client in client_info['active_clients']:
                try:
                    result['nodes'].append({
                        "id": str(client['_id']),
                        "label": "Client",
                        "role": client['role'],
                        "status": client['status'],
                        "name": client['name'],
                        "combiner": client['combiner'],
                        "type": 'client',
                    })
                except Exception as err:
                    print(err)

            count = 0
            for node in result['nodes']:
                try:
                    if node['type'] == 'combiner':
                        result['edges'].append(
                            {
                                "id": "e{}".format(count),
                                "source": node['id'],
                                "target": 'reducer',
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
                except Exception:
                    pass
                count = count + 1
            return result

        @app.route('/networkgraph')
        def network_graph():

            try:
                plot = Plot(self.control.statestore)
                result = netgraph()
                df_nodes = pd.DataFrame(result['nodes'])
                df_edges = pd.DataFrame(result['edges'])
                graph = plot.make_netgraph_plot(df_edges, df_nodes)
                return json.dumps(json_item(graph, "myplot"))
            except Exception:
                return ''

        @app.route('/events')
        def events():
            """

            :return:
            """

            json_docs = []
            for doc in self.control.get_events():
                json_doc = json.dumps(doc, default=json_util.default)
                json_docs.append(json_doc)

            json_docs.reverse()
            return {'events': json_docs}

        @app.route('/add')
        def add():
            """ Add a combiner to the network. """
            if self.token_auth_enabled:
                self.authorize(request, app.config.get('SECRET_KEY'))
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
                certificate, key = self.certificate_manager.get_or_create(
                    address).get_keypair_raw()
                _ = base64.b64encode(certificate)
                _ = base64.b64encode(key)

                # TODO append and redirect to index.

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
            """

            :return:
            """
            for r in request.headers:
                print("header contains: {}".format(r), flush=True)

            return render_template('eula.html', configured=True)

        @app.route('/models', methods=['GET', 'POST'])
        def models():
            """

            :return:
            """
            # Token auth
            if self.token_auth_enabled:
                self.authorize(request, app.config.get('SECRET_KEY'))

            if request.method == 'POST':
                # upload seed file
                uploaded_seed = request.files['seed']
                if uploaded_seed:

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
            """

            :return:
            """
            if request.method == 'POST':

                statestore_config = self.control.statestore.get_config()
                self.tracer = MongoTracer(
                    statestore_config['mongo_config'], statestore_config['network_id'])
                try:
                    self.control.drop_models()
                except Exception:
                    pass

                # drop objects in minio
                self.control.delete_bucket_objects()
                return redirect(url_for('models'))
            seed = True
            return redirect(url_for('models', seed=seed))

        @app.route('/drop_control', methods=['GET', 'POST'])
        def drop_control():
            """

            :return:
            """
            if request.method == 'POST':
                self.control.statestore.drop_control()
                return redirect(url_for('control'))
            return redirect(url_for('control'))

        # http://localhost:8090/control?rounds=4&model_id=879fa112-c861-4cb1-a25d-775153e5b548
        @app.route('/control', methods=['GET', 'POST'])
        def control():
            """ Main page for round control. Configure, start and stop global training rounds. """
            # Token auth
            if self.token_auth_enabled:
                self.authorize(request, app.config.get('SECRET_KEY'))

            not_configured = self.check_configured()
            if not_configured:
                return not_configured
            client = self.name
            state = ReducerStateToString(self.control.state())
            logs = None
            refresh = True

            if self.remote_compute_context:
                try:
                    self.current_compute_context = self.control.get_compute_context()
                except Exception:
                    self.current_compute_context = None
            else:
                self.current_compute_context = "None:Local"
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
                for combiner in self.control.network.get_combiners():
                    try:
                        combiner_state = combiner.report()
                        nac = combiner_state['nr_active_clients']
                        clients_available = clients_available + int(nac)
                    except Exception:
                        pass

                if clients_available < clients_required:
                    return redirect(url_for('index', state=state,
                                            message="Not enough clients available to start rounds! "
                                                    "check combiner client capacity",
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

                threading.Thread(target=self.control.instruct,
                                 args=(config,)).start()
                # self.control.instruct(config)
                return redirect(url_for('index', state=state, refresh=refresh, message="Sent execution plan.",
                                        message_type='SUCCESS'))

            else:
                seed_model_id = None
                latest_model_id = None
                try:
                    seed_model_id = self.control.get_first_model()[0]
                    latest_model_id = self.control.get_latest_model()
                except Exception:
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
            if self.token_auth_enabled:
                self.authorize(request, app.config.get('SECRET_KEY'))

            response = self.check_configured_response()

            if response:
                return response

            name = request.args.get('name', None)
            combiner_preferred = request.args.get('combiner', None)

            if combiner_preferred:
                combiner = self.control.find(combiner_preferred)
            else:
                combiner = self.control.find_available_combiner()

            if combiner is None:
                return jsonify({'status': 'retry',
                                'package': self.package,
                                'msg': "Failed to assign to a combiner, try again later."})

            client = {
                'name': name,
                'combiner_preferred': combiner_preferred,
                'combiner': combiner.name,
                'ip': request.remote_addr,
                'status': 'available'
            }

            # Add client to database
            self.control.network.add_client(client)

            # Return connection information to client

            cert_b64 = base64.b64encode(combiner.certificate)
            response = {
                'status': 'assigned',
                'host': combiner.address,
                'package': self.package,
                'ip': combiner.ip,
                'port': combiner.port,
                'certificate': str(cert_b64).split('\'')[1],
                'model_type': self.control.statestore.get_framework()
            }

            return jsonify(response)

        def combiner_status():
            """ Get current status reports from all combiners registered in the network.

            :return:
            """
            combiner_info = []
            for combiner in self.control.network.get_combiners():
                try:
                    report = combiner.report()
                    combiner_info.append(report)
                except Exception:
                    pass
            return combiner_info

        def client_status():
            """
            Get current status of clients (available) from DB compared with client status from all combiners,
            update client status to DB and add their roles.
            """
            client_info = self.control.network.get_client_info()
            combiner_info = combiner_status()
            try:
                all_active_trainers = []
                all_active_validators = []

                for client in combiner_info:
                    active_trainers_str = client['active_trainers']
                    active_validators_str = client['active_validators']
                    active_trainers_str = re.sub(
                        '[^a-zA-Z0-9-:\n\.]', '', active_trainers_str).replace('name:', ' ')  # noqa: W605
                    active_validators_str = re.sub(
                        '[^a-zA-Z0-9-:\n\.]', '', active_validators_str).replace('name:', ' ')  # noqa: W605
                    all_active_trainers.extend(
                        ' '.join(active_trainers_str.split(" ")).split())
                    all_active_validators.extend(
                        ' '.join(active_validators_str.split(" ")).split())

                active_trainers_list = [
                    client for client in client_info if client['name'] in all_active_trainers]
                active_validators_list = [
                    cl for cl in client_info if cl['name'] in all_active_validators]
                all_clients = [cl for cl in client_info]

                for client in all_clients:
                    status = 'offline'
                    role = 'None'
                    self.control.network.update_client_data(
                        client, status, role)

                all_active_clients = active_validators_list + active_trainers_list
                for client in all_active_clients:
                    status = 'active'
                    if client in active_trainers_list and client in active_validators_list:
                        role = 'trainer-validator'
                    elif client in active_trainers_list:
                        role = 'trainer'
                    elif client in active_validators_list:
                        role = 'validator'
                    else:
                        role = 'unknown'
                    self.control.network.update_client_data(
                        client, status, role)

                return {'active_clients': all_clients,
                        'active_trainers': active_trainers_list,
                        'active_validators': active_validators_list
                        }
            except Exception:
                pass

            return {'active_clients': [],
                    'active_trainers': [],
                    'active_validators': []
                    }

        @app.route('/metric_type', methods=['GET', 'POST'])
        def change_features():
            """

            :return:
            """
            feature = request.args['selected']
            plot = Plot(self.control.statestore)
            graphJSON = plot.create_box_plot(feature)
            return graphJSON

        @app.route('/dashboard')
        def dashboard():
            """

            :return:
            """
            # Token auth
            if self.token_auth_enabled:
                self.authorize(request, app.config.get('SECRET_KEY'))

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
            """

            :return:
            """
            # Token auth
            if self.token_auth_enabled:
                self.authorize(request, app.config.get('SECRET_KEY'))

            not_configured = self.check_configured()
            if not_configured:
                return not_configured
            plot = Plot(self.control.statestore)
            round_time_plot = plot.create_round_plot()
            mem_cpu_plot = plot.create_cpu_plot()
            combiners_plot = plot.create_combiner_plot()
            combiner_info = combiner_status()
            active_clients = client_status()
            return render_template('network.html', network_plot=True,
                                   round_time_plot=round_time_plot,
                                   mem_cpu_plot=mem_cpu_plot,
                                   combiners_plot=combiners_plot,
                                   combiner_info=combiner_info,
                                   active_clients=active_clients['active_clients'],
                                   active_trainers=active_clients['active_trainers'],
                                   active_validators=active_clients['active_validators'],
                                   configured=True
                                   )

        @app.route('/config/download', methods=['GET'])
        def config_download():
            """

            :return:
            """
            chk_string = ""
            name = self.control.get_compute_context()
            if name is None or name == '':
                chk_string = ''
            else:
                file_path = os.path.join(UPLOAD_FOLDER, name)
                print("trying to get {}".format(file_path))

                try:
                    sum = str(sha(file_path))
                except FileNotFoundError:
                    sum = ''
                chk_string = "checksum: {}".format(sum)

            network_id = self.network_id
            discover_host = self.name
            discover_port = self.port
            ctx = """network_id: {network_id}
controller:
    discover_host: {discover_host}
    discover_port: {discover_port}
    {chk_string}""".format(network_id=network_id,
                           discover_host=discover_host,
                           discover_port=discover_port,
                           chk_string=chk_string)

            obj = BytesIO()
            obj.write(ctx.encode('UTF-8'))
            obj.seek(0)
            return send_file(obj,
                             as_attachment=True,
                             attachment_filename='client.yaml',
                             mimetype='application/x-yaml')

        @app.route('/context', methods=['GET', 'POST'])
        def context():
            """

            :return:
            """
            # Token auth
            if self.token_auth_enabled:
                self.authorize(request, app.config.get('SECRET_KEY'))

            # if self.control.state() != ReducerState.setup or self.control.state() != ReducerState.idle:
            #    return "Error, Context already assigned!"
            # if reset is not empty then allow context re-set
            reset = request.args.get('reset', None)
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
                    file_path = os.path.join(
                        app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)

                    if self.control.state() == ReducerState.instructing or self.control.state() == ReducerState.monitoring:
                        return "Not allowed to change context while execution is ongoing."

                    self.control.set_compute_context(filename, file_path)
                    self.control.statestore.set_framework(helper_type)
                    return redirect(url_for('control'))

            name = request.args.get('name', '')

            if name == '':
                name = self.control.get_compute_context()
                if name is None or name == '':
                    return render_template('context.html')

            # There is a potential race condition here, if one client requests a package and at
            # the same time another one triggers a fetch from Minio and writes to disk.
            try:
                mutex = Lock()
                mutex.acquire()
                return send_from_directory(app.config['UPLOAD_FOLDER'], name, as_attachment=True)
            except Exception:
                try:
                    data = self.control.get_compute_package(name)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], name)
                    with open(file_path, 'wb') as fh:
                        fh.write(data)
                    return send_from_directory(app.config['UPLOAD_FOLDER'], name, as_attachment=True)
                except Exception:
                    raise
            finally:
                mutex.release()

            return render_template('context.html')

        @app.route('/checksum', methods=['GET', 'POST'])
        def checksum():
            """

            :return:
            """
            # sum = ''
            name = request.args.get('name', None)
            if name == '' or name is None:
                name = self.control.get_compute_context()
                if name is None or name == '':
                    return jsonify({})

            file_path = os.path.join(UPLOAD_FOLDER, name)
            print("trying to get {}".format(file_path))

            try:
                sum = str(sha(file_path))
            except FileNotFoundError:
                sum = ''

            data = {'checksum': sum}

            return jsonify(data)

        if self.certificate:
            print("Starting server with certs {} and key {}".format(str(self.certificate.cert_path),
                                                                    str(self.certificate.key_path)), flush=True)
            app.run(host="0.0.0.0", port=self.port,
                    ssl_context=(str(self.certificate.cert_path), str(self.certificate.key_path)))
        else:
            app.run(host="0.0.0.0", port=self.port)

        return app
