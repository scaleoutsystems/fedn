import base64
import copy
import json
import os
import threading
from io import BytesIO

from bson import json_util
from flask import jsonify, send_from_directory
from werkzeug.utils import secure_filename

from fedn.network.combiner.interfaces import (CombinerInterface,
                                              CombinerUnavailableError)
from fedn.network.state import ReducerState, ReducerStateToString
from fedn.utils.checksum import sha


class API:
    """ The API class is a wrapper for the statestore. It is used to expose the statestore to the network API. """

    def __init__(self, statestore, control):
        self.statestore = statestore
        self.control = control
        self.name = 'api'

    def _to_dict(self):
        """ Convert the object to a dict.
        return: The object as a dict.
        rtype: dict
        """
        data = {
            'name': self.name
        }
        return data

    def _get_combiner_report(self, combiner_id):
        """ Get report response from combiner.
        param: combiner_id: The combiner id to get report response from.
        type: combiner_id: str
        return: The report response from combiner.
        rtype: dict
        """
        # Get CombinerInterface (fedn.network.combiner.inferface.CombinerInterface) for combiner_id
        combiner = self.control.network.get_combiner(combiner_id)
        report = combiner.report
        return report

    def _allowed_file_extension(self, filename, ALLOWED_EXTENSIONS={'gz', 'bz2', 'tar', 'zip', 'tgz'}):
        """ Check if file extension is allowed.
        param: filename: The filename to check.
        type: filename: str
        return: True if file extension is allowed, else False.
        rtype: bool
        """

        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def get_all_clients(self):
        """ Get all clients from the statestore.
        return: All clients as a json object.
        rtype: json
        """
        # Will return list of ObjectId
        clients_objects = self.statestore.list_clients()
        payload = {}
        for object in clients_objects:
            id = object['name']
            info = {"combiner": object['combiner'],
                    "combiner_preferred": object['combiner_preferred'],
                    "ip": object['ip'],
                    "updated_at": object['updated_at'],
                    "status": object['status'],
                    }
            payload[id] = info

        return jsonify(payload)

    def get_active_clients(self, combiner_id):
        """ Get all active clients, i.e that are assigned to a combiner.
            A report request to the combiner is neccessary to determine if a client is active or not.
        param: combiner_id: The combiner id to get active clients for.
        type: combiner_id: str
        return: All active clients as a json object.
        rtype: json
        """
        # Get combiner interface object
        combiner = self.control.network.get_combiner(combiner_id)
        if combiner is None:
            return jsonify({"success": False, "message": f"Combiner {combiner_id}  not found."}), 404
        response = combiner.list_active_clients()
        return response

    def get_all_combiners(self):
        """ Get all combiners from the statestore.
        return: All combiners as a json object.
        rtype: json
        """
        # Will return list of ObjectId
        combiner_objects = self.statestore.get_combiners()
        payload = {}
        for object in combiner_objects:
            id = object['name']
            info = {"address": object['address'],
                    "fqdn": object['fqdn'],
                    "parent_reducer": object['parent']["name"],
                    "port": object['port'],
                    "report": object['report'],
                    "updated_at": object['updated_at'],
                    }
            payload[id] = info

        return jsonify(payload)

    def get_combiner(self, combiner_id):
        """ Get a combiner from the statestore.
        param: combiner_id: The combiner id to get.
        type: combiner_id: str
        return: The combiner info dict as a json object.
        rtype: json
        """
        # Will return ObjectId
        object = self.statestore.get_combiner(combiner_id)
        payload = {}
        id = object['name']
        info = {"address": object['address'],
                "fqdn": object['fqdn'],
                "parent_reducer": object['parent']["name"],
                "port": object['port'],
                "report": object['report'],
                "updated_at": object['updated_at'],
                }
        payload[id] = info

        return jsonify(payload)

    def get_all_sessions(self):
        """ Get all sessions from the statestore.
        return: All sessions as a json object.
        rtype: json
        """
        sessions_objects = self.statestore.get_sessions()
        if sessions_objects is None:
            return jsonify({"success": False, "message": "No sessions found."}), 404
        payload = {}
        for object in sessions_objects:
            id = object['session_id']
            info = object['session_config'][0]
            payload[id] = info
        return jsonify(payload)

    def get_session(self, session_id):
        """ Get a session from the statestore.
        param: session_id: The session id to get.
        type: session_id: str
        return: The session info dict as a json object.
        rtype: json
        """
        session_object = self.statestore.get_session(session_id)
        if session_object is None:
            return jsonify({"success": False, "message": f"Session {session_id} not found."}), 404
        payload = {}
        id = session_object['session_id']
        info = session_object['session_config'][0]
        payload[id] = info
        return jsonify(payload)

    def set_compute_package(self, file, helper_type):
        """ Set the compute package in the statestore.
        param: file: The compute package to set.
        type: file: file
        return: True if the compute package was set, else False.
        rtype: bool
        """

        if file and self._allowed_file_extension(file.filename):
            filename = secure_filename(file.filename)
            # TODO: make configurable, perhaps in config.py or package.py
            file_path = os.path.join(
                '/app/client/package/', filename)
            file.save(file_path)

            if self.control.state() == ReducerState.instructing or self.control.state() == ReducerState.monitoring:
                return jsonify({"success": False, "message": "Reducer is in instructing or monitoring state."
                                "Cannot set compute package."}), 400

            self.control.set_compute_package(filename, file_path)
            self.statestore.set_helper(helper_type)

        success = self.statestore.set_compute_package(filename)
        if not success:
            return jsonify({"success": False, "message": "Failed to set compute package."}), 400
        return jsonify({"success": True, "message": "Compute package set."})

    def _get_compute_package_name(self):
        """ Get the compute package name from the statestore.
        return: The compute package name.
        rtype: str
        """
        package_objects = self.statestore.get_compute_package()
        if package_objects is None:
            message = "No compute package found."
            return None, message
        else:
            try:
                name = package_objects['filename']
            except KeyError as e:
                message = "No compute package found. Key error."
                print(e)
                return None, message
            return name, 'success'

    def get_compute_package(self):
        """ Get the compute package from the statestore.
        return: The compute package as a json object.
        rtype: json
        """
        package_object = self.statestore.get_compute_package()
        if package_object is None:
            return jsonify({"success": False, "message": "No compute package found."}), 404
        payload = {}
        id = str(package_object['_id'])
        info = {"filename": package_object['filename'],
                "helper": package_object['helper'],
                }
        payload[id] = info
        return jsonify(payload)

    def download_compute_package(self, name):
        """ Download the compute package.
        return: The compute package as a json object.
        rtype: json
        """
        if name is None:
            name, message = self._get_compute_package_name()
            if name is None:
                return jsonify({"success": False, "message": message}), 404
        try:
            mutex = threading.Lock()
            mutex.acquire()
            # TODO: make configurable, perhaps in config.py or package.py
            return send_from_directory('/app/client/package/', name, as_attachment=True)
        except Exception:
            try:
                data = self.control.get_compute_package(name)
                # TODO: make configurable, perhaps in config.py or package.py
                file_path = os.path.join('/app/client/package/', name)
                with open(file_path, 'wb') as fh:
                    fh.write(data)
                # TODO: make configurable, perhaps in config.py or package.py
                return send_from_directory('/app/client/package/', name, as_attachment=True)
            except Exception:
                raise
        finally:
            mutex.release()

    def get_checksum(self, name):
        """ Get the checksum of the compute package.
        return: The checksum as a json object.
        rtype: json
        """

        if name is None:
            name, message = self._get_compute_package_name()
            if name is None:
                return jsonify({"success": False, "message": message}), 404

        file_path = os.path.join('/app/client/package/', name)  # TODO: make configurable, perhaps in config.py or package.py
        try:
            sum = str(sha(file_path))
        except FileNotFoundError:
            sum = ''

        payload = {'checksum': sum}

        return jsonify(payload)

    def get_controller_status(self):
        """ Get the status of the controller.
        return: The status of the controller as a json object.
        rtype: json
        """
        return jsonify({'state': ReducerStateToString(self.control.state())})
    # function with kwargs

    def get_events(self, **kwargs):
        """ Get the events of the federated network.
        return: The events as a json object.
        rtype: json
        """
        event_objects = self.statestore.get_events(**kwargs)
        if event_objects is None:
            return jsonify({"success": False, "message": "No events found."}), 404
        json_docs = []
        for doc in self.statestore.get_events(**kwargs):
            json_doc = json.dumps(doc, default=json_util.default)
            json_docs.append(json_doc)

        json_docs.reverse()
        return jsonify({'events': json_docs})

    def get_all_validations(self, **kwargs):
        """ Get all validations from the statestore.
        return: All validations as a json object.
        rtype: json
        """
        validations_objects = self.statestore.get_validations(**kwargs)
        if validations_objects is None:
            return jsonify(
                {
                    "success": False,
                    "message": "No validations found.",
                    "filter_used": kwargs
                }
            ), 404
        payload = {}
        for object in validations_objects:
            id = str(object['_id'])
            info = {
                'model_id': object['modelId'],
                'data': object['data'],
                'timestamp': object['timestamp'],
                'meta': object['meta'],
                'sender': object['sender'],
                'receiver': object['receiver'],
            }
            payload[id] = info
        return jsonify(payload)

    def add_combiner(self, combiner_id, secure_grpc, address, remote_addr, fqdn, port):
        """ Add a combiner to the network.
        param: combiner_id: The combiner id to add.
        type: combiner_id: str
        param: secure_grpc: Whether to use secure grpc or not.
        type: secure_grpc: bool
        param: name: The name of the combiner.
        type: name: str
        param: address: The address of the combiner.
        type: address: str
        param: remote_addr: The remote address of the combiner.
        type: remote_addr: str
        param: fqdn: The fqdn of the combiner.
        type: fqdn: str
        param: port: The port of the combiner.
        type: port: int
        return: Config of the combiner as a json object.
        rtype: json
        """
        # TODO: Any more required check for config? Formerly based on status: "retry"
        if not self.control.idle():
            return jsonify({
                'success': False,
                'status': 'retry',
                'message': 'Conroller is not in idle state, try again later. '
            }
            )
        # Check if combiner already exists
        combiner = self.control.network.get_combiner(combiner_id)
        if not combiner:
            if secure_grpc == 'True':
                certificate, key = self.certificate_manager.get_or_create(
                    address).get_keypair_raw()
                _ = base64.b64encode(certificate)
                _ = base64.b64encode(key)

            else:
                certificate = None
                key = None

            combiner_interface = CombinerInterface(
                parent=self._to_dict(),
                name=combiner_id,
                address=address,
                fqdn=fqdn,
                port=port,
                certificate=copy.deepcopy(certificate),
                key=copy.deepcopy(key),
                ip=remote_addr)

            self.control.network.add_combiner(combiner_interface)

        # Check combiner now exists
        combiner = self.control.network.get_combiner(combiner_id)
        if not combiner:
            return jsonify({'success': False, 'message': 'Combiner not added.'})

        payload = {
            'success': True,
            'message': 'Combiner added successfully.',
            'status': 'added',
            'storage': self.statestore.get_storage_backend(),
            'statestore': self.statestore.get_config(),
            'certificate': combiner.get_certificate(),
            'key': combiner.get_key()
        }

        return jsonify(payload)

    def add_client(self, client_id, preferred_combiner, remote_addr):
        """ Add a client to the network.
        param: client_id: The client id to add.
        type: client_id: str
        param: preferred_combiner: The preferred combiner for the client. 
        If None, the combiner will be chosen based on availability.
        type: preferred_combiner: str
        return: True if the client was added, else False. As json.
        rtype: json
        """
        # Check if package has been set
        package_object = self.statestore.get_compute_package()
        if package_object is None:
            return jsonify({'success': False, 'status': 'retry', 'message': 'No compute package found. Set package in controller.'}), 203

        # Assign client to combiner
        if preferred_combiner:
            combiner = self.control.network.get_combiner(preferred_combiner)
            if combiner is None:
                return jsonify({'success': False,
                                'message': f'Combiner {preferred_combiner} not found or unavailable.'})
        else:
            combiner = self.control.network.find_available_combiner()
            if combiner is None:
                return jsonify({'success': False,
                                'message': 'No combiner available.'})

        client_config = {
            'name': client_id,
            'combiner_preferred': preferred_combiner,
            'combiner': combiner.name,
            'ip': remote_addr,
            'status': 'available'
        }
        # Add client to network
        self.control.network.add_client(client_config)

        # Setup response containing information about the combiner for assinging the client
        if combiner.certificate:
            cert_b64 = base64.b64encode(combiner.certificate)
            cert = str(cert_b64).split('\'')[1]
        else:
            cert = None

        payload = {
            'status': 'assigned',
            'host': combiner.address,
            'fqdn': combiner.fqdn,
            'package': 'remote',  # TODO: Make this configurable
            'ip': combiner.ip,
            'port': combiner.port,
            'certificate': cert,
            'helper_type': self.control.statestore.get_helper()
        }
        print("Seding payload: ", payload, flush=True)

        return jsonify(payload)

    def get_initial_model(self):
        """ Get the initial model from the statestore.
        return: The initial model as a json object.
        rtype: json
        """
        model_id = self.statestore.get_initial_model()
        payload = {
            'model_id': model_id
        }
        return jsonify(payload)

    def set_initial_model(self, file):
        """ Add an initial model to the network.
        param: file: The initial model to add.
        type: file: file
        return: True if the initial model was added, else False.
        rtype: bool
        """
        try:
            object = BytesIO()
            object.seek(0, 0)
            file.seek(0)
            object.write(file.read())
            helper = self.control.get_helper()
            object.seek(0)
            model = helper.load(object)
            self.control.commit(file.filename, model)
        except Exception as e:
            print(e, flush=True)
            return jsonify({'success': False, 'message': e})

        return jsonify({'success': True, 'message': 'Initial model added successfully.'})

    def get_latest_model(self):
        """ Get the latest model from the statestore.
        return: The initial model as a json object.
        rtype: json
        """
        if self.statestore.get_latest_model():
            model_id = self.statestore.get_latest_model()
            payload = {
                'model_id': model_id
            }
            return jsonify(payload)
        else:
            return jsonify({'success': False, 'message': 'No initial model set.'})

    def get_model_trail(self):
        """ Get the model trail for a given session.
        param: session: The session id to get the model trail for.
        type: session: str
        return: The model trail for the given session as a json object.
        rtype: json
        """
        model_info = self.statestore.get_model_trail()
        if model_info:
            return jsonify(model_info)
        else:
            return jsonify({'success': False, 'message': 'No model trail available.'})

    def get_all_rounds(self):
        """ Get all rounds.
        return: The rounds as json object.
        rtype: json
        """
        rounds_objects = self.statestore.get_rounds()
        if rounds_objects is None:
            jsonify({'success': False, 'message': 'No rounds available.'})
        payload = {}
        for object in rounds_objects:
            id = object['round_id']
            info = {'reducer': object['reducer'],
                    'combiners': object['combiners'],
                    }
            payload[id] = info
        else:
            return jsonify(payload)

    def get_round(self, round_id):
        """ Get a round.
        param: round_id: The round id to get.
        type: round_id: str
        return: The round as json object.
        rtype: json
        """
        round_object = self.statestore.get_round(round_id)
        if round_object is None:
            return jsonify({'success': False, 'message': 'Round not found.'})
        payload = {
            'round_id': round_object['round_id'],
            'reducer': round_object['reducer'],
            'combiners': round_object['combiners'],
        }
        return jsonify(payload)

    def start_session(self, session_id, rounds=5, round_timeout=180, round_buffer_size=-1, delete_models=False,
                      validate=True, helper='keras', min_clients=1, requested_clients=8):
        """ Start a session.
        param: session_id: The session id to start.
        type: session_id: str
        param: rounds: The number of rounds to perform.
        type: rounds: int
        param: round_timeout: The round timeout to use in seconds.
        type: round_timeout: int
        param: round_buffer_size: The round buffer size to use.
        type: round_buffer_size: int
        param: delete_models: Whether to delete models after each round at combiner (save storage).
        type: delete_models: bool
        param: validate: Whether to validate the model after each round.
        type: validate: bool
        param: min_clients: The minimum number of clients required.
        type: min_clients: int
        param: requested_clients: The requested number of clients.
        type: requested_clients: int
        return: True if the session was started, else False. In json format.
        rtype: json
        """
        # Check if session already exists
        session = self.statestore.get_session(session_id)
        if session:
            return jsonify({'success': False, 'message': 'Session already exists.'})

        # Check if session is running
        if self.control.state() == ReducerState.monitoring:
            return jsonify({'success': False, 'message': 'A session is already running.'})

        # Check available clients per combiner
        clients_available = 0
        for combiner in self.control.network.get_combiners():
            try:
                combiner_state = combiner.report()
                nr_active_clients = combiner_state['nr_active_clients']
                clients_available = clients_available + int(nr_active_clients)
            except CombinerUnavailableError as e:
                # TODO: Handle unavailable combiner, stop session or continue?
                print("COMBINER UNAVAILABLE: {}".format(e), flush=True)
                continue

        if clients_available < min_clients:
            return jsonify({'success': False, 'message': 'Not enough clients available to start session.'})

        # Check if validate is string and convert to bool
        if isinstance(validate, str):
            if validate.lower() == 'true':
                validate = True
            else:
                validate = False

        # Get lastest model as initial model for session
        model_id = self.statestore.get_latest_model()

        # Setup session config
        session_config = {'round_timeout': round_timeout,
                          'model_id': model_id,
                          'rounds': rounds,
                          'delete_models_storage': delete_models,
                          'clients_required': min_clients,
                          'clients_requested': requested_clients,
                          'task': (''),
                          'validate': validate,
                          'helper_type': helper
                          }

        # Start session
        threading.Thread(target=self.control.session,
                         args=(session_config,)).start()

        # Return success response
        return jsonify({'success': True, 'message': 'Session started successfully.', "config": session_config})
