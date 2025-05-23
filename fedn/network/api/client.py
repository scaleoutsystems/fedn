import inspect
import os
import uuid

import requests

from fedn.network.combiner.hooks.serverfunctionsbase import ServerFunctionsBase

__all__ = ["APIClient"]


class APIClient:
    """An API client for interacting with the statestore and controller.

    :param host: The host of the api server.
    :type host: str
    :param port: The port of the api server.
    :type port: int
    :param secure: Whether to use https.
    :type secure: bool
    :param verify: Whether to verify the server certificate.
    :type verify: bool
    """

    def __init__(self, host: str, port: int = None, secure: bool = False, verify: bool = False, token: str = None, auth_scheme: str = None):
        if "://" in host:
            host = host.split("://")[1]
        self.host = host
        self.port = port
        self.secure = secure
        self.verify = verify
        self.headers = {}
        # Auth scheme passed as argument overrides environment variable.
        # "Token" is the default auth scheme.
        if not auth_scheme:
            auth_scheme = os.environ.get("FEDN_AUTH_SCHEME", "Bearer")
        # Override potential env variable if token is passed as argument.
        if not token:
            token = os.environ.get("FEDN_AUTH_TOKEN", False)

        if token:
            # Split the token if it contains a space (scheme + token).
            if " " in token:
                token = token.split()[1]
            self.headers = {"Authorization": f"{auth_scheme} {token}"}

    def _get_url(self, endpoint):
        if self.secure:
            protocol = "https"
        else:
            protocol = "http"
        if self.port:
            return f"{protocol}://{self.host}:{self.port}/{endpoint}"
        return f"{protocol}://{self.host}/{endpoint}"

    def _get_url_api_v1(self, endpoint):
        return self._get_url(f"api/v1/{endpoint}")

    # --- Clients --- #

    def get_client(self, id: str):
        """Get a client from the statestore.

        :param id: The client id to get.
        :type id: str
        :return: Client.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f"clients/{id}"), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_clients(self, n_max: int = None):
        """Get clients from the statestore.

        :param n_max: The maximum number of clients to get (If none all will be fetched).
        :type n_max: int
        return: Clients.
        rtype: dict
        """
        _headers = self.headers.copy()

        if n_max:
            _headers["X-Limit"] = str(n_max)

        response = requests.get(self._get_url_api_v1("clients/"), verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    def get_clients_count(self):
        """Get the number of clients in the statestore.

        :return: The number of clients.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("clients/count"), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_client_config(self, checksum=True):
        """Get client config from controller. Optionally include the checksum.
        The config is used for clients to connect to the controller and ask for combiner assignment.

        :param checksum: Whether to include the checksum of the package.
        :type checksum: bool
        :return: The client configuration.
        :rtype: dict
        """
        _params = {"checksum": "true" if checksum else "false"}

        response = requests.get(self._get_url_api_v1("clients/config"), params=_params, verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_active_clients(self, combiner_id: str = None, n_max: int = None):
        """Get active clients from the statestore.

        :param combiner_id: The combiner id to get active clients for.
        :type combiner_id: str
        :param n_max: The maximum number of clients to get (If none all will be fetched).
        :type n_max: int
        :return: Active clients.
        :rtype: dict
        """
        _params = {"status": "online"}

        if combiner_id:
            _params["combiner"] = combiner_id

        _headers = self.headers.copy()

        if n_max:
            _headers["X-Limit"] = str(n_max)

        response = requests.get(self._get_url_api_v1("clients/"), params=_params, verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    # --- Combiners --- #

    def get_combiner(self, id: str):
        """Get a combiner from the statestore.

        :param id: The combiner id to get.
        :type id: str
        :return: Combiner.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f"combiners/{id}"), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_combiners(self, n_max: int = None):
        """Get combiners in the network.

        :param n_max: The maximum number of combiners to get (If none all will be fetched).
        :type n_max: int
        :return: Combiners.
        :rtype: dict
        """
        _headers = self.headers.copy()

        if n_max:
            _headers["X-Limit"] = str(n_max)

        response = requests.get(self._get_url_api_v1("combiners/"), verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    def get_combiners_count(self):
        """Get the number of combiners in the statestore.

        :return: The number of combiners.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("combiners/count"), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    # --- Controllers --- #

    def get_controller_status(self):
        """Get the status of the controller.

        :return: The status of the controller.
        :rtype: dict
        """
        response = requests.get(self._get_url("get_controller_status"), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    # --- Models --- #

    def get_model(self, id: str):
        """Get a model from the statestore.

        :param id: The id (or model property) of the model to get.
        :type id: str
        :return: Model.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f"models/{id}"), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_models(self, session_id: str = None, n_max: int = None):
        """Get models from the statestore.

        :param session_id: The session id to get models for. (optional)
        :type session_id: str
        :param n_max: The maximum number of models to get (If none all will be fetched).
        :type n_max: int
        :return: Models.
        :rtype: dict
        """
        _params = {}

        if session_id:
            _params["session_id"] = session_id

        _headers = self.headers.copy()

        if n_max:
            _headers["X-Limit"] = str(n_max)

        response = requests.get(self._get_url_api_v1("models/"), params=_params, verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    def get_models_count(self):
        """Get the number of models in the statestore.

        :return: The number of models.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("models/count"), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_active_model(self):
        """Get the latest model from the statestore.

        :return: The latest model.
        :rtype: dict
        """
        _headers = self.headers.copy()
        _headers["X-Limit"] = "1"
        _headers["X-Sort-Key"] = "committed_at"
        _headers["X-Sort-Order"] = "desc"

        response = requests.get(self._get_url_api_v1("models/"), verify=self.verify, headers=_headers)
        _json = response.json()

        if "result" in _json and len(_json["result"]) > 0:
            return _json["result"][0]

        return _json

    def get_model_trail(self, id: str = None, include_self: bool = True, reverse: bool = True, n_max: int = None):
        """Get the model trail.

        :param id: The id (or model property) of the model to start the trail from. (optional)
        :type id: str
        :param n_max: The maximum number of models to get (If none all will be fetched).
        :type n_max: int
        :return: Models.
        :rtype: dict
        """
        if not id:
            model = self.get_active_model()
            if "model_id" in model:
                id = model["model_id"]
            else:
                return model

        _headers = self.headers.copy()

        _count: int = n_max if n_max else self.get_models_count()

        _headers["X-Limit"] = str(_count)
        _headers["X-Reverse"] = "true" if reverse else "false"

        _include_self_str: str = "true" if include_self else "false"

        response = requests.get(self._get_url_api_v1(f"models/{id}/ancestors?include_self={_include_self_str}"), verify=self.verify, headers=_headers)
        _json = response.json()

        return _json

    def download_model(self, id: str, path: str):
        """Download the model with id id.

        :param id: The id (or model property) of the model to download.
        :type id: str
        :param path: The path to download the model to.
        :type path: str
        :return: Message with success or failure.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f"models/{id}/download"), verify=self.verify, headers=self.headers)

        if response.status_code == 200:
            with open(path, "wb") as file:
                file.write(response.content)
            return {"success": True, "message": "Model downloaded successfully."}
        else:
            return {"success": False, "message": "Failed to download model."}

    def set_active_model(self, path):
        """Set the initial model in the statestore and upload to model repository.

        :param path: The file path of the initial model to set.
        :type path: str
        :return: A dict with success or failure message.
        :rtype: dict
        """
        if path.endswith(".npz"):
            helper = "numpyhelper"
        elif path.endswith(".bin"):
            helper = "binaryhelper"

        if helper:
            response = requests.put(self._get_url_api_v1("helpers/active"), json={"helper": helper}, verify=self.verify, headers=self.headers)

        with open(path, "rb") as file:
            response = requests.post(self._get_url_api_v1("models/"), files={"file": file}, data={"helper": helper}, verify=self.verify, headers=self.headers)
        return response.json()

    # --- Packages --- #

    def get_package(self, id: str):
        """Get a compute package from the statestore.

        :param id: The id of the compute package to get.
        :type id: str
        :return: Package.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f"packages/{id}"), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_packages(self, n_max: int = None):
        """Get compute packages from the statestore.

        :param n_max: The maximum number of packages to get (If none all will be fetched).
        :type n_max: int
        :return: Packages.
        :rtype: dict
        """
        _headers = self.headers.copy()

        if n_max:
            _headers["X-Limit"] = str(n_max)

        response = requests.get(self._get_url_api_v1("packages/"), verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    def get_packages_count(self):
        """Get the number of compute packages in the statestore.

        :return: The number of packages.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("packages/count"), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_active_package(self):
        """Get the (active) compute package from the statestore.

        :return: Package.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("packages/active"), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_package_checksum(self):
        """Get the checksum of the compute package.

        :return: The checksum.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("packages/checksum"), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def download_package(self, path: str):
        """Download the compute package.

        :param path: The path to download the compute package to.
        :type path: str
        :return: Message with success or failure.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("packages/download"), verify=self.verify, headers=self.headers)
        if response.status_code == 200:
            with open(path, "wb") as file:
                file.write(response.content)
            return {"success": True, "message": "Package downloaded successfully."}
        else:
            return {"success": False, "message": "Failed to download package."}

    def set_active_package(self, path: str, helper: str, name: str, description: str = ""):
        """Set the compute package in the statestore.

        :param path: The file path of the compute package to set.
        :type path: str
        :param helper: The helper type to use.
        :type helper: str
        :return: A dict with success or failure message.
        :rtype: dict
        """
        with open(path, "rb") as file:
            response = requests.post(
                self._get_url_api_v1("packages/"),
                files={"file": file},
                data={"helper": helper, "name": name, "description": description},
                verify=self.verify,
                headers=self.headers,
            )

        _json = response.json()

        return _json

    # --- Rounds --- #

    def get_round(self, id: str):
        """Get a round from the statestore.

        :param round_id: The round id to get.
        :type round_id: str
        :return: Round (config and metrics).
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f"rounds/{id}"), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_rounds(self, n_max: int = None):
        """Get all rounds from the statestore.

        :param n_max: The maximum number of rounds to get (If none all will be fetched).
        :type n_max: int
        :return: Rounds.
        :rtype: dict
        """
        _headers = self.headers.copy()

        if n_max:
            _headers["X-Limit"] = str(n_max)

        response = requests.get(self._get_url_api_v1("rounds/"), verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    def get_rounds_count(self):
        """Get the number of rounds in the statestore.

        :return: The number of rounds.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("rounds/count"), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    # --- Sessions --- #

    def get_session(self, id: str = None, name: str = None):
        """Get a session from the statestore.

        :param id: The session id to get.
        :type id: str
        :param name: The session name to get.
        :type name: str
        :return: Session.
        :rtype: dict
        """
        if name:
            response = requests.get(self._get_url_api_v1(f"sessions?name={name}"), verify=self.verify, headers=self.headers)
            _json = response.json()
            if "result" in _json and len(_json["result"]) > 0:
                _json = _json["result"][0]
            else:
                _json = {"message": "Session not found."}
        elif id:
            response = requests.get(self._get_url_api_v1(f"sessions/{id}"), verify=self.verify, headers=self.headers)
            _json = response.json()
        else:
            _json = {"message": "No id or name provided."}

        return _json

    def get_sessions(self, n_max: int = None, name: str = None):
        """Get sessions from the statestore.

        :param n_max: The maximum number of sessions to get (If none all will be fetched).
        :type n_max: int
        :param name: The session name to get.
        :type name: str
        :return: Sessions.
        :rtype: dict
        """
        _headers = self.headers.copy()

        if n_max:
            _headers["X-Limit"] = str(n_max)

        url = self._get_url_api_v1("sessions/")
        if name:
            url += f"?name={name}"

        response = requests.get(url, verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    def get_sessions_count(self):
        """Get the number of sessions in the statestore.

        :return: The number of sessions.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("sessions/count"), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_session_status(self, id: str):
        """Get the status of a session.

        :param id: The id of the session to get.
        :type id: str
        :return: The status of the session.
        :rtype: str
        """
        session = self.get_session(id)

        if session and "status" in session:
            return session["status"]

        return "Could not retrieve session status."

    def session_is_finished(self, id: str):
        """Check if a session with id has finished.

        :param id: The id of the session to get.
        :type id: str
        :return: True if session is finished, otherwise false.
        :rtype: bool
        """
        status = self.get_session_status(id)
        return status and status.lower() == "finished"

    def start_session(
        self,
        name: str = None,
        aggregator: str = "fedavg",
        aggregator_kwargs: dict = None,
        model_id: str = None,
        round_timeout: int = 180,
        rounds: int = 5,
        round_buffer_size: int = -1,
        delete_models: bool = True,
        validate: bool = True,
        helper: str = None,
        min_clients: int = 1,
        requested_clients: int = 8,
        server_functions: ServerFunctionsBase = None,
    ):
        """Start a new session.

        :param name: The name of the session
        :type name: str
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
        is_splitlearning = aggregator == "splitlearningagg"
        if is_splitlearning:
            return self.start_splitlearning_session(
                name, model_id, round_timeout, rounds, round_buffer_size, delete_models, validate, min_clients, requested_clients
            )

        if model_id is None:
            _headers = self.headers.copy()
            _headers["X-Limit"] = "1"
            _headers["X-Sort-Key"] = "committed_at"
            _headers["X-Sort-Order"] = "desc"
            response = requests.get(self._get_url_api_v1("models/"), verify=self.verify, headers=_headers)
            if response.status_code == 200:
                json = response.json()
                if "result" in json and len(json["result"]) > 0:
                    model_id = json["result"][0]["model_id"]
                else:
                    return {"message": "No models found in the repository"}
            else:
                return {"message": "No models found in the repository"}

        if helper is None:
            response = requests.get(self._get_url_api_v1("helpers/active"), verify=self.verify, headers=self.headers)
            if response.status_code == 400:
                helper = "numpyhelper"
            elif response.status_code == 200:
                helper = response.json()
            else:
                return {"message": "An unexpected error occurred when getting the active helper"}

        response = requests.post(
            self._get_url_api_v1("sessions/"),
            json={
                "name": name,
                "seed_model_id": model_id,
                "session_config": {
                    "aggregator": aggregator,
                    "aggregator_kwargs": aggregator_kwargs,
                    "round_timeout": round_timeout,
                    "buffer_size": round_buffer_size,
                    "model_id": model_id,
                    "delete_models_storage": delete_models,
                    "clients_required": min_clients,
                    "requested_clients": requested_clients,
                    "validate": validate,
                    "helper_type": helper,
                    "server_functions": None if server_functions is None else inspect.getsource(server_functions),
                },
            },
            verify=self.verify,
            headers=self.headers,
        )

        if response.status_code == 201:
            session_id = response.json()["session_id"]
            response = requests.post(
                self._get_url_api_v1("sessions/start"),
                json={
                    "session_id": session_id,
                    "rounds": rounds,
                    "round_timeout": round_timeout,
                },
                verify=self.verify,
                headers=self.headers,
            )
            # Try to parse JSON, but handle the case where it fails
            try:
                response_json = response.json()
                response_json["session_id"] = session_id
                return response_json
            except requests.exceptions.JSONDecodeError:
                # Handle invalid JSON response
                return {"success": response.status_code < 400, "session_id": session_id, "message": f"Session started with status code {response.status_code}"}

        _json = response.json()
        return _json

    def start_splitlearning_session(
        self,
        name: str = None,
        model_id: str = None,
        round_timeout: int = 180,
        rounds: int = 5,
        round_buffer_size: int = -1,
        delete_models: bool = True,
        validate: bool = False,
        min_clients: int = 1,
        requested_clients: int = 8,
    ):
        """Start a new splitlearning session.

        :param name: The name of the session
        :type name: str
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
        :param min_clients: The minimum number of clients required.
        :type min_clients: int
        :param requested_clients: The requested number of clients.
        :type requested_clients: int
        :return: A dict with success or failure message and session config.
        :rtype: dict
        """
        response = requests.post(
            self._get_url_api_v1("sessions/"),
            json={
                "name": name,
                "session_config": {
                    "aggregator": "splitlearningagg",
                    "model_id": model_id,
                    "round_timeout": round_timeout,
                    "buffer_size": round_buffer_size,
                    "delete_models_storage": delete_models,
                    "clients_required": min_clients,
                    "requested_clients": requested_clients,
                    "validate": validate,
                    "helper_type": "splitlearninghelper",
                },
            },
            verify=self.verify,
            headers=self.headers,
        )

        if response.status_code == 201:
            session_id = response.json()["session_id"]
            response = requests.post(
                self._get_url_api_v1("sessions/start_splitlearning_session"),
                json={
                    "session_id": session_id,
                    "rounds": rounds,
                    "round_timeout": round_timeout,
                },
                verify=self.verify,
                headers=self.headers,
            )
            response_json = response.json()
            response_json["session_id"] = session_id
            return response_json

        _json = response.json()
        return _json

    def continue_session(self, session_id: str, rounds: int = 5, round_timeout: int = 180):
        """Continue a session.

        :param session_id: The id of the session to continue.
        :type session_id: str
        :param rounds: The number of rounds to perform.
        :type rounds: int
        :param round_timeout: The round timeout to use in seconds.
        :type round_timeout: int
        :return: A dict with success or failure message and session config.
        :rtype: dict
        """
        if not session_id:
            return {"message": "No session id provided."}
        if rounds is None or rounds <= 0:
            return {"message": "Invalid number of rounds provided. Must be greater than 0."}
        if round_timeout is None or round_timeout <= 0:
            return {"message": "Invalid round timeout provided. Must be greater than 0."}
        # Check if session exists
        session = self.get_session(session_id)
        if not session or "session_id" not in session:
            return {"message": "Session not found."}
        # Check if session is finished
        if not self.session_is_finished(session_id):
            return {"message": "Session is already running."}

        response = requests.post(
            self._get_url_api_v1("sessions/start"),
            json={
                "session_id": session_id,
                "rounds": rounds,
                "round_timeout": round_timeout,
            },
            verify=self.verify,
            headers=self.headers,
        )

        _json = response.json()

        return _json

    # --- Statuses --- #

    def get_status(self, id: str):
        """Get a status object (event) from the statestore.

        :param id: The id of the status to get.
        :type id: str
        :return: Status.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f"statuses/{id}"), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    def get_statuses(self, session_id: str = None, event_type: str = None, sender_name: str = None, sender_role: str = None, n_max: int = None):
        """Get statuses from the statestore. Filter by input parameters

        :param session_id: The session id to get statuses for.
        :type session_id: str
        :param event_type: The event type to get.
        :type event_type: str
        :param sender_name: The sender name to get.
        :type sender_name: str
        :param sender_role: The sender role to get.
        :type sender_role: str
        :param n_max: The maximum number of statuses to get (If none all will be fetched).
        :type n_max: int
        :return: Statuses
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
            _headers["X-Limit"] = str(n_max)

        response = requests.get(self._get_url_api_v1("statuses/"), params=_params, verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    def get_statuses_count(self):
        """Get the number of statuses in the statestore.

        :return: The number of statuses.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("statuses/count"), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    # --- Validations --- #

    def get_validation(self, id: str):
        """Get a validation from the statestore.

        :param id: The id of the validation to get.
        :type id: str
        :return: Validation.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1(f"validations/{id}"), verify=self.verify, headers=self.headers)

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
        n_max: int = None,
    ):
        """Get validations from the statestore. Filter by input parameters.

        :param session_id: The session id to get validations for.
        :type session_id: str
        :param model_id: The model id to get validations for.
        :type model_id: str
        :param correlation_id: The correlation id to get validations for.
        :type correlation_id: str
        :param sender_name: The sender name to get validations for.
        :type sender_name: str
        :param sender_role: The sender role to get validations for.
        :type sender_role: str
        :param receiver_name: The receiver name to get validations for.
        :type receiver_name: str
        :param receiver_role: The receiver role to get validations for.
        :type receiver_role: str
        :param n_max: The maximum number of validations to get (If none all will be fetched).
        :type n_max: int
        :return: Validations.
        :rtype: dict
        """
        _params = {}

        if session_id:
            _params["sessionId"] = session_id

        if model_id:
            _params["modelId"] = model_id

        if correlation_id:
            _params["correlationId"] = correlation_id

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
            _headers["X-Limit"] = str(n_max)

        response = requests.get(self._get_url_api_v1("validations/"), params=_params, verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    def get_validations_count(self):
        """Get the number of validations in the statestore.

        :return: The number of validations.
        :rtype: dict
        """
        response = requests.get(self._get_url_api_v1("validations/count"), verify=self.verify, headers=self.headers)

        _json = response.json()

        return _json

    # --- Predictions --- #

    def get_predictions(
        self,
        model_id: str = None,
        correlation_id: str = None,
        sender_name: str = None,
        sender_role: str = None,
        receiver_name: str = None,
        receiver_role: str = None,
        n_max: int = None,
    ):
        """Get predictions from the statestore. Filter by input parameters.

        :param model_id: The model id to get predictions for.
        :type model_id: str
        :param correlation_id: The correlation id to get predictions for.
        :type correlation_id: str
        :param sender_name: The sender name to get predictions for.
        :type sender_name: str
        :param sender_role: The sender role to get predictions for.
        :type sender_role: str
        :param receiver_name: The receiver name to get predictions for.
        :type receiver_name: str
        :param receiver_role: The receiver role to get predictions for.
        :type receiver_role: str
        :param n_max: The maximum number of predictions to get (If none all will be fetched).
        :type n_max: int
        :return: Predictions.
        :rtype: dict
        """
        _params = {}

        if model_id:
            _params["modelId"] = model_id

        if correlation_id:
            _params["correlationId"] = correlation_id

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
            _headers["X-Limit"] = str(n_max)

        response = requests.get(self._get_url_api_v1("predictions/"), params=_params, verify=self.verify, headers=_headers)

        _json = response.json()

        return _json

    def start_predictions(self, prediction_id: str = None, model_id: str = None):
        """Start predictions for a model.

        :param model_id: The model id to start predictions for.
        :type model_id: str
        :param data: The data to predict.
        :type data: dict
        :return: A dict with success or failure message.
        :rtype: dict
        """
        if not prediction_id:
            prediction_id = str(uuid.uuid4())
        response = requests.post(
            self._get_url_api_v1("predictions/start"),
            json={"prediction_id": prediction_id, "model_id": model_id},
            verify=self.verify,
            headers=self.headers,
        )

        _json = response.json()

        return _json

    # --- Client --- #

    def get_current_attributes(self, client_list):
        """Get the current attributes of the client.

        :param client_list: The list of clients to get the attributes for or a single client_id
        :type client_list: list|str
        :raises ValueError: If client_list is not a list or empty.
        :return: The current attributes of the client.
        :rtype: dict
        """
        if not isinstance(client_list, list):
            if isinstance(client_list, str):
                client_list = [client_list]
            else:
                raise ValueError("client_list must be a list")
        if len(client_list) == 0:
            raise ValueError("client_list must not be empty")
        json = {"client_ids": client_list}
        response = requests.post(self._get_url_api_v1("attributes/current"), json=json, verify=self.verify, headers=self.headers)
        _json = response.json()
        return _json

    def add_attributes(self, attribute: dict) -> dict:
        """Add or update client attributes via the controller API.

        :param attribute: A dict matching AttributeDTO.schema, e.g.
            {
                "key": "charging",
                "value": "true",
                "sender": {"name": "", "role": "", "client_id": "abc123"}
            }
        :return: Parsed JSON response from the server.
        :rtype: dict
        """
        url = self._get_url_api_v1("attributes/")
        response = requests.post(url, json=attribute, headers=self.headers, verify=self.verify)
        response.raise_for_status()
        return response.json()
