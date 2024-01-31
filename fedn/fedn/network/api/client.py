import json

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

    def __init__(self, host, port, secure=False, verify=False):
        self.host = host
        self.port = port
        self.secure = secure
        self.verify = verify

    def _get_url(self, endpoint):
        if self.secure:
            protocol = 'https'
        else:
            protocol = 'http'
        return f'{protocol}://{self.host}:{self.port}/{endpoint}'

    def get_model_trail(self):
        """ Get the model trail.

        :return: The model trail as dict including commit timestamp.
        :rtype: dict
        """
        response = requests.get(self._get_url('get_model_trail'), verify=self.verify)
        return response.json()

    def list_models(self, session_id=None):
        """ Get all models from the statestore.

        :return: All models.
        :rtype: dict
        """
        response = requests.get(self._get_url('list_models'), params={'session_id': session_id}, verify=self.verify)
        return response.json()

    def list_clients(self):
        """ Get all clients from the statestore.

        return: All clients.
        rtype: dict
        """
        response = requests.get(self._get_url('list_clients'))
        return response.json()

    def get_active_clients(self, combiner_id):
        """ Get all active clients from the statestore.

        :param combiner_id: The combiner id to get active clients for.
        :type combiner_id: str
        :return: All active clients.
        :rtype: dict
        """
        response = requests.get(self._get_url('get_active_clients'), params={'combiner': combiner_id}, verify=self.verify)
        return response.json()

    def get_client_config(self, checksum=True):
        """ Get the controller configuration. Optionally include the checksum.
        The config is used for clients to connect to the controller and ask for combiner assignment.

        :param checksum: Whether to include the checksum of the package.
        :type checksum: bool
        :return: The client configuration.
        :rtype: dict
        """
        response = requests.get(self._get_url('get_client_config'), params={'checksum': checksum}, verify=self.verify)
        return response.json()

    def list_combiners(self):
        """ Get all combiners in the network.

        :return: All combiners with info.
        :rtype: dict
        """
        response = requests.get(self._get_url('list_combiners'))
        return response.json()

    def get_combiner(self, combiner_id):
        """ Get a combiner from the statestore.

        :param combiner_id: The combiner id to get.
        :type combiner_id: str
        :return: The combiner info.
        :rtype: dict
        """
        response = requests.get(self._get_url(f'get_combiner?combiner={combiner_id}'), verify=self.verify)
        return response.json()

    def list_rounds(self):
        """ Get all rounds from the statestore.

        :return: All rounds with config and metrics.
        :rtype: dict
        """
        response = requests.get(self._get_url('list_rounds'))
        return response.json()

    def get_round(self, round_id):
        """ Get a round from the statestore.

        :param round_id: The round id to get.
        :type round_id: str
        :return: The round config and metrics.
        :rtype: dict
        """
        response = requests.get(self._get_url(f'get_round?round_id={round_id}'), verify=self.verify)
        return response.json()

    def start_session(self, session_id=None, aggregator='fedavg', model_id=None, round_timeout=180, rounds=5, round_buffer_size=-1, delete_models=True,
                      validate=True, helper='numpyhelper', min_clients=1, requested_clients=8):
        """ Start a new session.

        :param session_id: The session id to start.
        :type session_id: str
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
            'session_id': session_id,
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
        }, verify=self.verify
        )
        return response.json()

    def list_sessions(self):
        """ Get all sessions from the statestore.

        :return: All sessions in dict.
        :rtype: dict
        """
        response = requests.get(self._get_url('list_sessions'), verify=self.verify)
        return response.json()

    def get_session(self, session_id):
        """ Get a session from the statestore.

        :param session_id: The session id to get.
        :type session_id: str
        :return: The session as a json object.
        :rtype: dict
        """
        response = requests.get(self._get_url(f'get_session?session_id={session_id}'), self.verify)
        return response.json()

    def session_is_finished(self, session_id):
        """ Check if a session with id session_id has finished.

        :param session_id: The session id to get.
        :type session_id: str
        :return: The session as a json object.
        :rtype: dict
        """
        try:
            status = self.get_session(session_id)['status']
            if status == 'Finished':
                return True
            else:
                return False
        except json.JSONDecodeError:
            # Could happen if the session has not been written to db yet
            return False
        except Exception:
            raise

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
                                     'helper': helper, 'name': name, 'description': description}, verify=self.verify)
        return response.json()

    def get_package(self):
        """ Get the compute package from the statestore.

        :return: The compute package with info.
        :rtype: dict
        """
        response = requests.get(self._get_url('get_package'), verify=self.verify)
        return response.json()

    def list_compute_packages(self):
        """ Get all compute packages from the statestore.

        :return: All compute packages with info.
        :rtype: dict
        """
        response = requests.get(self._get_url('list_compute_packages'), verify=self.verify)
        return response.json()

    def download_package(self, path):
        """ Download the compute package.

        :param path: The path to download the compute package to.
        :type path: str
        :return: Message with success or failure.
        :rtype: dict
        """
        response = requests.get(self._get_url('download_package'), verify=self.verify)
        if response.status_code == 200:
            with open(path, 'wb') as file:
                file.write(response.content)
            return {'success': True, 'message': 'Package downloaded successfully.'}
        else:
            return {'success': False, 'message': 'Failed to download package.'}

    def get_package_checksum(self):
        """ Get the checksum of the compute package.

        :return: The checksum.
        :rtype: dict
        """
        response = requests.get(self._get_url('get_package_checksum'), verify=self.verify)
        return response.json()

    def get_latest_model(self):
        """ Get the latest model from the statestore.

        :return: The latest model id.
        :rtype: dict
        """
        response = requests.get(self._get_url('get_latest_model'), verify=self.verify)
        return response.json()

    def get_initial_model(self):
        """ Get the initial model from the statestore.

        :return: The initial model id.
        :rtype: dict
        """
        response = requests.get(self._get_url('get_initial_model'), verify=self.verify)
        return response.json()

    def set_initial_model(self, path):
        """ Set the initial model in the statestore and upload to model repository.

        :param path: The file path of the initial model to set.
        :type path: str
        :return: A dict with success or failure message.
        :rtype: dict
        """
        with open(path, 'rb') as file:
            response = requests.post(self._get_url('set_initial_model'), files={'file': file}, verify=self.verify)
        return response.json()

    def get_controller_status(self):
        """ Get the status of the controller.

        :return: The status of the controller.
        :rtype: dict
        """
        response = requests.get(self._get_url('get_controller_status'), verify=self.verify)
        return response.json()

    def get_events(self, **kwargs):
        """ Get the events from the statestore. Pass kwargs to filter events.

        :return: The events in dict
        :rtype: dict
        """
        response = requests.get(self._get_url('get_events'), params=kwargs, verify=self.verify)
        return response.json()

    def list_validations(self, **kwargs):
        """ Get all validations from the statestore. Pass kwargs to filter validations.

        :return: All validations in dict.
        :rtype: dict
        """
        response = requests.get(self._get_url('list_validations'), params=kwargs, verify=self.verify)
        return response.json()
