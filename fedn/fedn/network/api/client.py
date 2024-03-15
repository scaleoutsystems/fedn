import os

import requests

__all__ = ['APIClient']


class APIClient:
    """ An API client for interacting with the statestore and controller.

    :param host: The host of the api server.
    :type host: str
    :param port: The port of the api server.
    :type port: int
    :param secure: Whether to use https.
    :type secure: bool
    :param verify: Whether to verify the server certificate.
    :type verify: bool
    """

    def __init__(self, host, port=None, secure=False, verify=False, token=None, auth_scheme=None):
        self.host = host
        self.port = port
        self.secure = secure
        self.verify = verify
        self.headers = {}
        # Auth scheme passed as argument overrides environment variable.
        # "Token" is the default auth scheme.
        if not auth_scheme:
            auth_scheme = os.environ.get("FEDN_AUTH_SCHEME", "Token")
        # Override potential env variable if token is passed as argument.
        if not token:
            token = os.environ.get("FEDN_AUTH_TOKEN", False)

        if token:
            self.headers = {"Authorization": f"{auth_scheme} {token}"}

    def _get_url(self, endpoint):
        if self.secure:
            protocol = 'https'
        else:
            protocol = 'http'
        if self.port:
            return f'{protocol}://{self.host}:{self.port}/{endpoint}'
        return f'{protocol}://{self.host}/{endpoint}'

    def _get_url_api_v1(self, endpoint):
        return self._get_url(f'api/v1/{endpoint}')

    # --- Clients --- #

    def get_client(self, id: str):
        """ Get a client from the statestore.

        :param id: The client id to get.
        :type id: str
        :return: The client info.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f'clients/{id}'), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_clients(self, n_max: int = None):
        """ Get all clients from the statestore.

        return: All clients.
        rtype: dict
        """
        _headers = self.headers.copy()

        if n_max:
            _headers['X-Limit'] = str(n_max)

        response = requests.get(self._get_url_api_v1('clients'), verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    def get_active_clients(self, combiner_id: str = None, n_max: int = None):
        """ Get all active clients from the statestore.

        :param combiner_id: The combiner id to get active clients for.
        :type combiner_id: str
        :return: All active clients.
        :rtype: dict
        """
        _params = {"status": "online"}

        if combiner_id:
            _params['combiner'] = combiner_id

        _headers = self.headers.copy()

        if n_max:
            _headers['X-Limit'] = str(n_max)

        response = requests.get(self._get_url_api_v1('clients'), params=_params, verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    # --- Combiners --- #

    def get_combiner(self, id: str):
        """ Get a combiner from the statestore.

        :param id: The combiner id to get.
        :type id: str
        :return: The combiner info.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f'combiners/{id}'), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_combiners(self, n_max: int = None):
        """ Get all combiners in the network.

        :return: All combiners with info.
        :rtype: dict
        """
        _headers = self.headers.copy()

        if n_max:
            _headers['X-Limit'] = str(n_max)

        response = requests.get(self._get_url_api_v1('combiners'), verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    # --- Controller --- #

    def get_controller_config(self, checksum=True):
        """ Get the controller configuration. Optionally include the checksum.
        The config is used for clients to connect to the controller and ask for combiner assignment.

        :param checksum: Whether to include the checksum of the package.
        :type checksum: bool
        :return: The client configuration.
        :rtype: dict
        """
        response = requests.get(self._get_url('get_client_config'), params={'checksum': checksum}, verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_controller_status(self):
        """ Get the status of the controller.

        :return: The status of the controller.
        :rtype: dict
        """
        response = requests.get(self._get_url('get_controller_status'), verify=self.verify, headers=self.headers)
        
        _json = response.json()

        return _json

    # --- Models --- #

    def get_model(self, id: str):
        """ Get a model from the statestore.

        :param id: The model id to get.
        :type id: str
        :return: The model info.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f'models/{id}'), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_models(self, session_id: str = None, n_max: int = None):
        """ Get all models from the statestore.

        :return: All models.
        :rtype: dict
        """
        _params = {}

        if session_id:
            _params['session_id'] = session_id

        _headers = self.headers.copy()

        if n_max:
            _headers['X-Limit'] = str(n_max)

        response = requests.get(self._get_url_api_v1('models'), params=_params, verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    def get_active_model(self):
        """ Get the latest model from the statestore.

        :return: The latest model id.
        :rtype: dict
        """
        _headers = self.headers.copy()
        _headers['X-Limit'] = "1"

        response = requests.get(self._get_url_api_v1('models'), verify=self.verify, headers=_headers)
        _json = response.json()

        if "result" in _json and len(_json["result"]) > 0:
            return _json["result"][0]

        return _json

    def get_model_trail(self, id: str = None, n_max: int = None):
        """ Get the model trail.

        :return: The model trail as dict including commit timestamp.
        :rtype: dict
        """
        if not id:
            model = self.get_latest_model()
            if "id" in model:
                id = model["id"]
            else:
                return model

        _headers = self.headers.copy()

        if n_max:
            _headers['X-Limit'] = str(n_max)

        response = requests.get(self._get_url_api_v1(f'models/{id}/ancestors'), verify=self.verify, headers=self.headers)
        _json = response.json()

        return _json

    def set_model(self, path):
        """ Set the initial model in the statestore and upload to model repository.

        :param path: The file path of the initial model to set.
        :type path: str
        :return: A dict with success or failure message.
        :rtype: dict
        """
        with open(path, 'rb') as file:
            response = requests.post(self._get_url('set_initial_model'), files={'file': file}, verify=self.verify, headers=self.headers)
        return response.json()

    # --- Packages --- #

    def get_package(self, id: str):
        """ Get a compute package from the statestore.

        :param id: The compute package id to get.
        :type id: str
        :return: The compute package with info.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f'packages/{id}'), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_packages(self, n_max: int = None):
        """ Get all compute packages from the statestore.

        :return: All compute packages with info.
        :rtype: dict
        """
        _headers = self.headers.copy()

        if n_max:
            _headers['X-Limit'] = str(n_max)

        response = requests.get(self._get_url_api_v1('packages'), verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    def get_active_package(self):
        """ Get the compute package from the statestore.

        :return: The compute package with info.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1('packages/active'), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_package_checksum(self):
        """ Get the checksum of the compute package.

        :return: The checksum.
        :rtype: dict
        """
        response = requests.get(self._get_url('get_package_checksum'), verify=self.verify, headers=self.headers)
        
        _json = response.json()

        return _json

    def download_package(self, path: str):
        """ Download the compute package.

        :param path: The path to download the compute package to.
        :type path: str
        :return: Message with success or failure.
        :rtype: dict
        """
        response = requests.get(self._get_url('download_package'), verify=self.verify, headers=self.headers)
        if response.status_code == 200:
            with open(path, 'wb') as file:
                file.write(response.content)
            return {'success': True, 'message': 'Package downloaded successfully.'}
        else:
            return {'success': False, 'message': 'Failed to download package.'}

    def set_package(self, path: str, helper: str, name: str = None, description: str = None):
        """ Set the compute package in the statestore.

        :param path: The file path of the compute package to set.
        :type path: str
        :param helper: The helper type to use.
        :type helper: str
        :return: A dict with success or failure message.
        :rtype: dict
        """
        with open(path, 'rb') as file:
            response = requests.post(self._get_url('set_package'), files={'file': file}, data={
                                     'helper': helper, 'name': name, 'description': description}, verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    # --- Rounds --- #

    def get_round(self, id: str):
        """ Get a round from the statestore.

        :param round_id: The round id to get.
        :type round_id: str
        :return: The round config and metrics.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f'rounds/{id}'), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_rounds(self, n_max: int = None):
        """ Get all rounds from the statestore.

        :return: All rounds with config and metrics.
        :rtype: dict
        """
        _headers = self.headers.copy()

        if n_max:
            _headers['X-Limit'] = str(n_max)

        response = requests.get(self._get_url_api_v1('rounds'), verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    # --- Sessions --- #

    def get_session(self, id: str):
        """ Get a session from the statestore.

        :param id: The session id to get.
        :type id: str
        :return: The session as a json object.
        :rtype: dict
        """

        response = requests.get(self._get_url_api_v1(f'sessions/{id}'), self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_sessions(self, n_max: int = None):
        """ Get all sessions from the statestore.

        :return: All sessions in dict.
        :rtype: dict
        """
        _headers = self.headers.copy()

        if n_max:
            _headers['X-Limit'] = str(n_max)

        response = requests.get(self._get_url_api_v1('sessions'), verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    def get_session_status(self, id: str):
        """ Check if a session with id id has finished.

        :param id: The session id to get.
        :type id: str
        :return: The session as a json object.
        :rtype: dict
        """
        session = self.get_session(id)

        if session and "status" in session:
            return session["status"]

        return "Could not retrieve session status."

    def get_session_is_finished(self, id: str):
        """ Check if a session with id id has finished.

        :param id: The session id to get.
        :type id: str
        :return: The session as a json object.
        :rtype: dict
        """
        status = self.get_session_status(id)
        return status and status.lower() == "finished"

    def start_session(self, id: str = None, aggregator: str = 'fedavg', model_id:  str = None, round_timeout: int = 180, rounds: int = 5, round_buffer_size: int = -1, delete_models: bool = True,
                      validate: bool = True, helper: str = 'numpyhelper', min_clients: int = 1, requested_clients: int = 8):
        """ Start a new session.

        :param id: The session id to start.
        :type id: str
        :param aggregator: The aggregator plugin to use.
        :type aggregator: str
        :param model_id: The id of the initial model.
        :type model_id: str
        :param round_timeout: The round timeout to use in seconds.
        :type round_timeout: int
        :param rounds: The number of rounds to perform.
        :type rounds: int
        :param round_buffer_size: The round buffer size to use.
        :type round_buffer_size: int
        :param delete_models: Whether to delete models after each round at combiner (save storage).
        :type delete_models: bool
        :param validate: Whether to validate the model after each round.
        :type validate: bool
        :param helper: The helper type to use.
        :type helper: str
        :param min_clients: The minimum number of clients required.
        :type min_clients: int
        :param requested_clients: The requested number of clients.
        :type requested_clients: int
        :return: A dict with success or failure message and session config.
        :rtype: dict
        """
        response = requests.post(self._get_url('start_session'), json={
            'id': id,
            'aggregator': aggregator,
            'model_id': model_id,
            'round_timeout': round_timeout,
            'rounds': rounds,
            'round_buffer_size': round_buffer_size,
            'delete_models': delete_models,
            'validate': validate,
            'helper': helper,
            'min_clients': min_clients,
            'requested_clients': requested_clients
        }, verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    # --- Statuses --- #

    def get_status(self, id: str):
        """ Get an event from the statestore.

        :param id: The event id to get.
        :type id: str
        :return: The event in dict.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f'statuses/{id}'), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_statuses(self, session_id: str = None, event_type: str = None, sender_name: str = None, sender_role: str = None, n_max: int = None):
        """ Get the events from the statestore. Pass kwargs to filter events.

        :return: The events in dict
        :rtype: dict
        """
        _params = {}

        if session_id:
            _params["session_id"] = session_id

        if event_type:
            _params["type"] = event_type

        if sender_name:
            _params["sender.name"] = sender_name

        if sender_role:
            _params["sender.role"] = sender_role

        _headers = self.headers.copy()

        if n_max:
            _headers['X-Limit'] = str(n_max)        

        response = requests.get(self._get_url_api_v1('statuses'), params=_params, verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    # --- Validations --- #

    def get_validation(self, id: str):
        """ Get a validation from the statestore.

        :param id: The validation id to get.
        :type id: str
        :return: The validation in dict.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f'validations/{id}'), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_validations(
        self,
        session_id: str = None,
        model_id: str = None,
        correlation_id: str = None,
        sender_name: str = None,
        sender_role: str = None,
        receiver_name: str = None,
        receiver_role: str = None,
        n_max: int = None
    ):
        """ Get all validations from the statestore. Pass kwargs to filter validations.

        :return: All validations in dict.
        :rtype: dict
        """
        _params = {}

        if session_id:
            _params["session_id"] = session_id

        if model_id:
            _params["model_id"] = model_id

        if correlation_id:
            _params["correlation_id"] = correlation_id

        if sender_name:
            _params["sender.name"] = sender_name
        
        if sender_role:
            _params["sender.role"] = sender_role
        
        if receiver_name:
            _params["receiver.name"] = receiver_name
        
        if receiver_role:
            _params["receiver.role"] = receiver_role

        _headers = self.headers.copy()

        if n_max:
            _headers['X-Limit'] = str(n_max)

        response = requests.get(self._get_url_api_v1('validations'), params=_params, verify=self.verify, headers=_headers)
        
        _json = response.json()

        return _json
