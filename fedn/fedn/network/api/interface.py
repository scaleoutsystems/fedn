import json
import threading

from bson import json_util
from flask import jsonify

from fedn.network.state import ReducerStateToString


class API:
    """ The API class is a wrapper for the statestore. It is used to expose the statestore to the network API. """

    def __init__(self, statestore, control):
        self.statestore = statestore
        self.control = control

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

    def set_compute_package(self, file):
        """ Set the compute package in the statestore.
        param: file: The compute package to set.
        type: file: file
        return: True if the compute package was set, else False.
        rtype: bool
        """
        success = self.statestore.set_compute_package(file)
        if not success:
            return jsonify({"success": False, "message": "Failed to set compute package."}), 400
        return jsonify({"success": True, "message": "Compute package set."})

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
        json_docs = []
        for doc in self.statestore.get_events(**kwargs):
            json_doc = json.dumps(doc, default=json_util.default)
            json_docs.append(json_doc)

        json_docs.reverse()
        return jsonify({'events': json_docs})

    def add_combiner(self, combiner_id, secure_grpc, name, address, remote_addr, fqdn, port):
        """ Add a combiner to the network.
        param: combiner_id: The combiner id to add.
        type: combiner_id: str
        return: True if the combiner was added, else False.
        rtype: bool
        """
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
                self,
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
            'storage': self.control.statestore.get_storage_backend(),
            'statestore': self.control.statestore.get_config(),
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

        return jsonify(payload)

    def get_initial_model(self):
        """ Get the initial model from the statestore.
        return: The initial model as a json object.
        rtype: json
        """
        model_id = self.control.get_first_model()
        payload = {
            'success': True,
            'message': 'Initial model retrieved successfully.',
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
        if file:
            object = BytesIO()
            object.seek(0, 0)
            file.seek(0)
            object.write(file.read())
            helper = self.control.get_helper()
            object.seek(0)
            model = helper.load(object)
            self.control.commit(file.filename, model)
        else:
            return jsonify({'success': False, 'message': 'No file provided.'})

        return jsonify({'success': True, 'message': 'Initial model added successfully.'})

    def get_latest_model(self):
        """ Get the latest model from the statestore.
        return: The initial model as a json object.
        rtype: json
        """
        if self.control.get_latest_model():
            model_id = self.control.get_latest_model()
            payload = {
                'success': True,
                'message': 'Initial model retrieved successfully.',
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
        model_info = self.statestore.get_model_info()
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
        #session = self.control.get_session(session_id)
        # if session:
        #    return jsonify({'success': False, 'message': 'Session already exists.'})

        # Check if session is running
        # if self.control.state() == ReducerState.RUNNING:
        #    return jsonify({'success': False, 'message': 'A session is already running.'})

        # Check available clients per combiner
        clients_available = 0
        for combiner in self.control.network.get_combiners():
            try:
                combiner_state = combiner.report()
                nr_active_clients = combiner_state['nr_active_clients']
                clients_available = clients_available + int(nr_active_clients)
            except Exception:
                pass
        if clients_available < min_clients:
            return jsonify({'success': False, 'message': 'Not enough clients available to start session.'})

        # Check if validate is string and convert to bool
        if isinstance(validate, str):
            if validate.lower() == 'true':
                validate = True
            else:
                validate = False

        # Get lastest model as initial model for session
        model_id = self.control.get_latest_model()

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
