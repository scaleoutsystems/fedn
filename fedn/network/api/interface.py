import os
import threading
import uuid
from io import BytesIO

from flask import jsonify, send_from_directory
from werkzeug.security import safe_join
from werkzeug.utils import secure_filename

from fedn.common.config import FEDN_COMPUTE_PACKAGE_DIR, get_controller_config, get_network_config
from fedn.common.log_config import logger
from fedn.network.combiner.interfaces import CombinerUnavailableError
from fedn.network.state import ReducerState, ReducerStateToString
from fedn.utils.checksum import sha

__all__ = ("API",)


class API:
    """The API class is a wrapper for the statestore. It is used to expose the statestore to the network API."""

    def __init__(self, statestore, control):
        self.statestore = statestore
        self.control = control
        self.name = "api"

    def _to_dict(self):
        """Convert the object to a dict.

        ::return: The object as a dict.
        ::rtype: dict
        """
        data = {"name": self.name}
        return data

    def _allowed_file_extension(self, filename, ALLOWED_EXTENSIONS={"gz", "bz2", "tar", "zip", "tgz"}):
        """Check if file extension is allowed.

        :param filename: The filename to check.
        :type filename: str
        :return: True and extension str if file extension is allowed, else False and None.
        :rtype: Tuple (bool, str)
        """
        if "." in filename:
            extension = filename.rsplit(".", 1)[1].lower()
            if extension in ALLOWED_EXTENSIONS:
                return (True, extension)

        return (False, None)

    def get_clients(self, limit=None, skip=None, status=False):
        """Get all clients from the statestore.

        :return: All clients as a json response.
        :rtype: :class:`flask.Response`
        """
        # Will return list of ObjectId
        response = self.statestore.list_clients(limit, skip, status)

        arr = []

        for element in response["result"]:
            obj = {
                "id": element["name"],
                "combiner": element["combiner"],
                "combiner_preferred": element["combiner_preferred"],
                "ip": element["ip"],
                "status": element["status"],
                "last_seen": element["last_seen"] if "last_seen" in element else "",
            }

            arr.append(obj)

        result = {"result": arr, "count": response["count"]}

        return jsonify(result)

    def get_all_combiners(self, limit=None, skip=None):
        """Get all combiners from the statestore.

        :return: All combiners as a json response.
        :rtype: :class:`flask.Response`
        """
        # Will return list of ObjectId
        projection = {"name": True, "updated_at": True}
        response = self.statestore.get_combiners(limit, skip, projection=projection)
        arr = []
        for element in response["result"]:
            obj = {
                "name": element["name"],
                "updated_at": element["updated_at"],
            }

            arr.append(obj)

        result = {"result": arr, "count": response["count"]}

        return jsonify(result)

    def get_combiner(self, combiner_id):
        """Get a combiner from the statestore.

        :param combiner_id: The combiner id to get.
        :type combiner_id: str
        :return: The combiner info dict as a json response.
        :rtype: :class:`flask.Response`
        """
        # Will return ObjectId
        object = self.statestore.get_combiner(combiner_id)
        payload = {}
        id = object["name"]
        info = {
            "address": object["address"],
            "fqdn": object["fqdn"],
            "parent_reducer": object["parent"]["name"],
            "port": object["port"],
            "updated_at": object["updated_at"],
        }
        payload[id] = info

        return jsonify(payload)

    def get_all_sessions(self, limit=None, skip=None):
        """Get all sessions from the statestore.

        :return: All sessions as a json response.
        :rtype: :class:`flask.Response`
        """
        sessions_object = self.statestore.get_sessions(limit, skip)
        if sessions_object is None:
            return (
                jsonify({"success": False, "message": "No sessions found."}),
                404,
            )
        arr = []
        for element in sessions_object["result"]:
            obj = element["session_config"][0]
            arr.append(obj)

        result = {"result": arr, "count": sessions_object["count"]}

        return jsonify(result)

    def get_session(self, session_id):
        """Get a session from the statestore.

        :param session_id: The session id to get.
        :type session_id: str
        :return: The session info dict as a json response.
        :rtype: :class:`flask.Response`
        """
        session_object = self.statestore.get_session(session_id)
        if session_object is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Session {session_id} not found.",
                    }
                ),
                404,
            )
        payload = {}
        id = session_object["session_id"]
        info = session_object["session_config"][0]
        status = session_object["status"]
        payload[id] = info
        payload["status"] = status
        return jsonify(payload)

    def set_active_compute_package(self, id: str):
        success = self.statestore.set_active_compute_package(id)

        if not success:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Failed to set compute package.",
                    }
                ),
                400,
            )

        return jsonify({"success": True, "message": "Compute package set."})

    def set_compute_package(self, file, helper_type: str, name: str = None, description: str = None):
        """Set the compute package in the statestore.

        :param file: The compute package to set.
        :type file: file
        :return: A json response with success or failure message.
        :rtype: :class:`flask.Response`
        """
        if self.control.state() == ReducerState.instructing or self.control.state() == ReducerState.monitoring:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Reducer is in instructing or monitoring state." "Cannot set compute package.",
                    }
                ),
                400,
            )

        if file is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "No file provided.",
                    }
                ),
                404,
            )

        success, extension = self._allowed_file_extension(file.filename)

        if not success:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"File extension {extension} not allowed.",
                    }
                ),
                404,
            )

        file_name = file.filename
        storage_file_name = secure_filename(f"{str(uuid.uuid4())}.{extension}")

        file_path = safe_join(FEDN_COMPUTE_PACKAGE_DIR, storage_file_name)
        file.save(file_path)

        self.control.set_compute_package(storage_file_name, file_path)
        success = self.statestore.set_compute_package(file_name, storage_file_name, helper_type, name, description)

        if not success:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Failed to set compute package.",
                    }
                ),
                400,
            )
        # Delete the file after it has been saved
        os.remove(file_path)
        return jsonify({"success": True, "message": "Compute package set."})

    def _get_compute_package_name(self):
        """Get the compute package name from the statestore.

        :return: The compute package name.
        :rtype: str
        """
        package_objects = self.statestore.get_compute_package()
        if package_objects is None:
            message = "No compute package found."
            return None, message
        else:
            try:
                name = package_objects["storage_file_name"]
            except KeyError as e:
                message = "No compute package found. Key error."
                logger.debug(e)
                return None, message
            return name, "success"

    def get_compute_package(self):
        """Get the compute package from the statestore.

        :return: The compute package as a json response.
        :rtype: :class:`flask.Response`
        """
        result = self.statestore.get_compute_package()
        if result is None:
            return (
                jsonify({"success": False, "message": "No compute package found."}),
                404,
            )

        obj = {
            "id": result["id"] if "id" in result else "",
            "file_name": result["file_name"],
            "helper": result["helper"],
            "committed_at": result["committed_at"],
            "storage_file_name": result["storage_file_name"] if "storage_file_name" in result else "",
            "name": result["name"] if "name" in result else "",
            "description": result["description"] if "description" in result else "",
        }

        return jsonify(obj)

    def list_compute_packages(self, limit: str = None, skip: str = None, include_active: str = None):
        """Get paginated list of compute packages from the statestore.
        :param limit: The number of compute packages to return.
        :type limit: str
        :param skip: The number of compute packages to skip.
        :type skip: str
        :param include_active: Whether to include the active compute package or not.
        :type include_active: str
        :return: All compute packages as a json response.
        :rtype: :class:`flask.Response`
        """
        if limit is not None and skip is not None:
            limit = int(limit)
            skip = int(skip)

        include_active: bool = include_active == "true"

        result = self.statestore.list_compute_packages(limit, skip)
        if result is None:
            return (
                jsonify({"success": False, "message": "No compute packages found."}),
                404,
            )

        active_package_id: str = None

        if include_active:
            active_package = self.statestore.get_compute_package()

            if active_package is not None:
                active_package_id = active_package["id"] if "id" in active_package else ""

        if include_active:
            arr = [
                {
                    "id": element["id"] if "id" in element else "",
                    "file_name": element["file_name"],
                    "helper": element["helper"],
                    "committed_at": element["committed_at"],
                    "storage_file_name": element["storage_file_name"] if "storage_file_name" in element else "",
                    "name": element["name"] if "name" in element else "",
                    "description": element["description"] if "description" in element else "",
                    "active": "id" in element and element["id"] == active_package_id,
                }
                for element in result["result"]
            ]
        else:
            arr = [
                {
                    "id": element["id"] if "id" in element else "",
                    "file_name": element["file_name"],
                    "helper": element["helper"],
                    "committed_at": element["committed_at"],
                    "storage_file_name": element["storage_file_name"] if "storage_file_name" in element else "",
                    "name": element["name"] if "name" in element else "",
                    "description": element["description"] if "description" in element else "",
                }
                for element in result["result"]
            ]

        result = {"result": arr, "count": result["count"]}
        return jsonify(result)

    def download_compute_package(self, name):
        """Download the compute package.

        :return: The compute package as a json object.
        :rtype: :class:`flask.Response`
        """
        if name is None:
            name, message = self._get_compute_package_name()
            if name is None:
                return jsonify({"success": False, "message": message}), 404
        try:
            mutex = threading.Lock()
            mutex.acquire()

            return send_from_directory(FEDN_COMPUTE_PACKAGE_DIR, name, as_attachment=True)
        except Exception:
            try:
                data = self.control.get_compute_package(name)
                # TODO: make configurable, perhaps in config.py or package.py
                file_path = safe_join(FEDN_COMPUTE_PACKAGE_DIR, name)
                with open(file_path, "wb") as fh:
                    fh.write(data)
                # TODO: make configurable, perhaps in config.py or package.py
                return send_from_directory(FEDN_COMPUTE_PACKAGE_DIR, name, as_attachment=True)
            except Exception:
                raise
        finally:
            mutex.release()

    def _create_checksum(self, name=None):
        """Create the checksum of the compute package.

        :param name: The name of the compute package.
        :type name: str
        :return: Success or failure boolean, message and the checksum.
        :rtype: bool, str, str
        """
        if name is None:
            name, message = self._get_compute_package_name()
            if name is None:
                return False, message, ""
        file_path = safe_join(os.getcwd(), name)  # TODO: make configurable, perhaps in config.py or package.py
        try:
            sum = str(sha(file_path))
        except FileNotFoundError:
            sum = ""
            message = "File not found."
        return True, message, sum

    def get_checksum(self, name):
        """Get the checksum of the compute package.

        :param name: The name of the compute package.
        :type name: str
        :return: The checksum as a json object.
        :rtype: :py:class:`flask.Response`
        """
        success, message, sum = self._create_checksum(name)
        if not success:
            return jsonify({"success": False, "message": message}), 404
        payload = {"checksum": sum}

        return jsonify(payload)

    def get_controller_status(self):
        """Get the status of the controller.

        :return: The status of the controller as a json object.
        :rtype: :py:class:`flask.Response`
        """
        return jsonify({"state": ReducerStateToString(self.control.state())})

    def get_events(self, **kwargs):
        """Get the events of the federated network.

        :return: The events as a json object.
        :rtype: :py:class:`flask.Response`
        """
        response = self.statestore.get_events(**kwargs)

        result = response["result"]
        if result is None:
            return (
                jsonify({"success": False, "message": "No events found."}),
                404,
            )

        events = []
        for evt in result:
            events.append(evt)

        return jsonify({"result": events, "count": response["count"]})

    def get_all_validations(self, **kwargs):
        """Get all validations from the statestore.

        :return: All validations as a json response.
        :rtype: :class:`flask.Response`
        """
        validations_objects = self.statestore.get_validations(**kwargs)
        if validations_objects is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "No validations found.",
                        "filter_used": kwargs,
                    }
                ),
                404,
            )
        payload = {}
        for object in validations_objects:
            id = str(object["_id"])
            info = {
                "model_id": object["modelId"],
                "data": object["data"],
                "timestamp": object["timestamp"],
                "meta": object["meta"],
                "sender": object["sender"],
                "receiver": object["receiver"],
            }
            payload[id] = info
        return jsonify(payload)

    def add_combiner(self, combiner_id, secure_grpc, address, remote_addr, fqdn, port):
        """Add a combiner to the network.

        :param combiner_id: The combiner id to add.
        :type combiner_id: str
        :param secure_grpc: Whether to use secure grpc or not.
        :type secure_grpc: bool
        :param name: The name of the combiner.
        :type name: str
        :param address: The address of the combiner.
        :type address: str
        :param remote_addr: The remote address of the combiner.
        :type remote_addr: str
        :param fqdn: The fqdn of the combiner.
        :type fqdn: str
        :param port: The port of the combiner.
        :type port: int
        :return: Config of the combiner as a json response.
        :rtype: :class:`flask.Response`
        """
        payload = {
            "success": False,
            "message": "Adding combiner via REST API is obsolete. Include statestore and object store config in combiner config.",
            "status": "abort",
        }

        return jsonify(payload)

    def add_client(self, client_id, preferred_combiner, remote_addr, name, package):
        """Add a client to the network.

        :param client_id: The client id to add.
        :type client_id: str
        :param preferred_combiner: The preferred combiner for the client.If None, the combiner will be chosen based on availability.
        :type preferred_combiner: str
        :return: A json response with combiner assignment config.
        :rtype: :class:`flask.Response`
        """
        if package == "remote":
            package_object = self.statestore.get_compute_package()
            if package_object is None:
                return (
                    jsonify(
                        {
                            "success": False,
                            "status": "retry",
                            "message": "No compute package found. Set package in controller.",
                        }
                    ),
                    203,
                )
            helper_type = self.control.statestore.get_helper()
        else:
            # Else package is "local":
            helper_type = ""

        # Assign client to combiner
        if preferred_combiner:
            combiner = self.control.network.get_combiner(preferred_combiner)
            if combiner is None:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Combiner {preferred_combiner} not found or unavailable.",
                        }
                    ),
                    400,
                )
        else:
            combiner = self.control.network.find_available_combiner()
            if combiner is None:
                return (
                    jsonify({"success": False, "message": "No combiner available."}),
                    400,
                )

        client_config = {
            "client_id": client_id,
            "name": name,
            "combiner_preferred": preferred_combiner,
            "combiner": combiner.name,
            "ip": remote_addr,
            "status": "available",
            "package": package,
        }
        # Add client to network
        self.control.network.add_client(client_config)

        payload = {
            "status": "assigned",
            "host": combiner.address,
            "fqdn": combiner.fqdn,
            "package": package,
            "ip": combiner.ip,
            "port": combiner.port,
            "helper_type": helper_type,
        }
        return jsonify(payload)

    def get_initial_model(self):
        """Get the initial model from the statestore.

        :return: The initial model as a json response.
        :rtype: :class:`flask.Response`
        """
        model_id = self.statestore.get_initial_model()
        payload = {"model_id": model_id}
        return jsonify(payload)

    def set_initial_model(self, file):
        """Add an initial model to the network.

        :param file: The initial model to add.
        :type file: file
        :return: A json response with success or failure message.
        :rtype: :class:`flask.Response`
        """
        logger.info("Adding model")
        try:
            object = BytesIO()
            object.seek(0, 0)
            file.seek(0)
            object.write(file.read())
            helper = self.control.get_helper()
            logger.info(f"Loading model from file using helper {helper.name}")
            object.seek(0)
            model = helper.load(object)
            self.control.commit(file.filename, model)
        except Exception as e:
            logger.error("Error occured during model loading")
            logger.debug(e)
            status_code = 400
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Failed to add model.",
                    }
                ),
                status_code,
            )

        return jsonify({"success": True, "message": "Initial model added successfully."})

    def get_latest_model(self):
        """Get the latest model from the statestore.

        :return: The initial model as a json response.
        :rtype: :class:`flask.Response`
        """
        if self.statestore.get_latest_model():
            model_id = self.statestore.get_latest_model()
            payload = {"model_id": model_id}
            return jsonify(payload)
        else:
            return jsonify({"success": False, "message": "No initial model set."})

    def set_current_model(self, model_id: str):
        """Set the active model in the statestore.

        :param model_id: The model id to set.
        :type model_id: str
        :return: A json response with success or failure message.
        :rtype: :class:`flask.Response`
        """
        success = self.statestore.set_current_model(model_id)

        if not success:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Failed to set active model.",
                    }
                ),
                400,
            )

        return jsonify({"success": True, "message": "Active model set."})

    def get_models(self, session_id: str = None, limit: str = None, skip: str = None, include_active: str = None):
        result = self.statestore.list_models(session_id, limit, skip)

        if result is None:
            return (
                jsonify({"success": False, "message": "No models found."}),
                404,
            )

        include_active: bool = include_active == "true"

        if include_active:
            latest_model = self.statestore.get_latest_model()

            arr = [
                {
                    "committed_at": element["committed_at"],
                    "model": element["model"],
                    "session_id": element["session_id"],
                    "active": element["model"] == latest_model,
                }
                for element in result["result"]
            ]
        else:
            arr = [
                {
                    "committed_at": element["committed_at"],
                    "model": element["model"],
                    "session_id": element["session_id"],
                }
                for element in result["result"]
            ]

        result = {"result": arr, "count": result["count"]}

        return jsonify(result)

    def get_model(self, model_id: str):
        result = self.statestore.get_model(model_id)

        if result is None:
            return (
                jsonify({"success": False, "message": "No model found."}),
                404,
            )

        payload = {
            "committed_at": result["committed_at"],
            "parent_model": result["parent_model"],
            "model": result["model"],
            "session_id": result["session_id"],
        }

        return jsonify(payload)

    def get_model_trail(self):
        """Get the model trail for a given session.

        :param session: The session id to get the model trail for.
        :type session: str
        :return: The model trail for the given session as a json response.
        :rtype: :class:`flask.Response`
        """
        model_info = self.statestore.get_model_trail()
        if model_info:
            return jsonify(model_info)
        else:
            return jsonify({"success": False, "message": "No model trail available."})

    def get_model_ancestors(self, model_id: str, limit: str = None):
        """Get the model ancestors for a given model.

        :param model_id: The model id to get the model ancestors for.
        :type model_id: str
        :param limit: The number of ancestors to return.
        :type limit: str
        :return: The model ancestors for the given model as a json response.
        :rtype: :class:`flask.Response`
        """
        if model_id is None:
            return jsonify({"success": False, "message": "No model id provided."})

        limit: int = int(limit) if limit is not None else 10  # if limit is None, default to 10

        response = self.statestore.get_model_ancestors(model_id, limit)
        if response:
            arr: list = []

            for element in response:
                obj = {
                    "model": element["model"],
                    "committed_at": element["committed_at"],
                    "session_id": element["session_id"],
                    "parent_model": element["parent_model"],
                }
                arr.append(obj)

            result = {"result": arr}

            return jsonify(result)
        else:
            return jsonify({"success": False, "message": "No model ancestors available."})

    def get_model_descendants(self, model_id: str, limit: str = None):
        """Get the model descendants for a given model.

        :param model_id: The model id to get the model descendants for.
        :type model_id: str
        :param limit: The number of descendants to return.
        :type limit: str
        :return: The model descendants for the given model as a json response.
        :rtype: :class:`flask.Response`
        """
        if model_id is None:
            return jsonify({"success": False, "message": "No model id provided."})

        limit: int = int(limit) if limit is not None else 10

        response: list = self.statestore.get_model_descendants(model_id, limit)

        if response:
            arr: list = []

            for element in response:
                obj = {
                    "model": element["model"],
                    "committed_at": element["committed_at"],
                    "session_id": element["session_id"],
                    "parent_model": element["parent_model"],
                }
                arr.append(obj)

            result = {"result": arr}

            return jsonify(result)
        else:
            return jsonify({"success": False, "message": "No model descendants available."})

    def get_all_rounds(self):
        """Get all rounds.

        :return: The rounds as json response.
        :rtype: :class:`flask.Response`
        """
        rounds_objects = self.statestore.get_rounds()
        if rounds_objects is None:
            jsonify({"success": False, "message": "No rounds available."})
        payload = {}
        for object in rounds_objects:
            id = object["round_id"]
            if "reducer" in object.keys():
                reducer = object["reducer"]
            else:
                reducer = None
            if "combiners" in object.keys():
                combiners = object["combiners"]
            else:
                combiners = None

            info = {
                "reducer": reducer,
                "combiners": combiners,
            }
            payload[id] = info
        return jsonify(payload)

    def get_round(self, round_id):
        """Get a round.

        :param round_id: The round id to get.
        :type round_id: str
        :return: The round as json response.
        :rtype: :class:`flask.Response`
        """
        round_object = self.statestore.get_round(round_id)
        if round_object is None:
            return jsonify({"success": False, "message": "Round not found."})
        payload = {
            "round_id": round_object["round_id"],
            "combiners": round_object["combiners"],
        }
        return jsonify(payload)

    def get_client_config(self, checksum=True):
        """Get the client config.

        :return: The client config as json response.
        :rtype: :py:class:`flask.Response`
        """
        config = get_controller_config()
        network_id = get_network_config()
        port = config["port"]
        host = config["host"]
        payload = {
            "network_id": network_id,
            "discover_host": host,
            "discover_port": port,
        }
        if checksum:
            success, _, checksum_str = self._create_checksum()
            if success:
                payload["checksum"] = checksum_str
        return jsonify(payload)

    def list_combiners_data(self, combiners):
        """Get combiners data.

        :param combiners: The combiners to get data for.
        :type combiners: list
        :return: The combiners data as json response.
        :rtype: :py:class:`flask.Response`
        """
        response = self.statestore.list_combiners_data(combiners)

        arr = []

        # order list by combiner name
        for element in response:
            obj = {
                "combiner": element["_id"],
                "count": element["count"],
            }

            arr.append(obj)

        result = {"result": arr}

        return jsonify(result)

    def start_session(
        self,
        session_id,
        aggregator="fedavg",
        aggregator_kwargs=None,
        model_id=None,
        rounds=5,
        round_timeout=180,
        round_buffer_size=-1,
        delete_models=True,
        validate=True,
        helper="",
        min_clients=1,
        requested_clients=8,
    ):
        """Start a session.

        :param session_id: The session id to start.
        :type session_id: str
        :param aggregator: The aggregator plugin to use.
        :type aggregator: str
        :param initial_model: The initial model for the session.
        :type initial_model: str
        :param rounds: The number of rounds to perform.
        :type rounds: int
        :param round_timeout: The round timeout to use in seconds.
        :type round_timeout: int
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
        :return: A json response with success or failure message and session config.
        :rtype: :class:`flask.Response`
        """
        # Check if session already exists
        session = self.statestore.get_session(session_id)
        if session:
            return jsonify({"success": False, "message": "Session already exists."})

        # Check if session is running
        if self.control.state() == ReducerState.monitoring:
            return jsonify({"success": False, "message": "A session is already running."})

        # Check if compute package is set
        package = self.statestore.get_compute_package()
        if not package:
            return jsonify(
                {
                    "success": False,
                    "message": "No compute package set. Set compute package before starting session.",
                }
            )
        if not helper:
            # get helper from compute package
            helper = package["helper"]

        # Check that initial (seed) model is set
        if not self.statestore.get_initial_model():
            return jsonify(
                {
                    "success": False,
                    "message": "No initial model set. Set initial model before starting session.",
                }
            )

        # Check available clients per combiner
        clients_available = 0
        for combiner in self.control.network.get_combiners():
            try:
                nr_active_clients = len(combiner.list_active_clients())
                clients_available = clients_available + int(nr_active_clients)
            except CombinerUnavailableError as e:
                # TODO: Handle unavailable combiner, stop session or continue?
                logger.error("COMBINER UNAVAILABLE: {}".format(e))
                continue

        if clients_available < min_clients:
            return jsonify(
                {
                    "success": False,
                    "message": "Not enough clients available to start session.",
                }
            )

        # Check if validate is string and convert to bool
        if isinstance(validate, str):
            if validate.lower() == "true":
                validate = True
            else:
                validate = False

        # Get lastest model as initial model for session
        if not model_id:
            model_id = self.statestore.get_latest_model()

        # Setup session config
        session_config = {
            "session_id": session_id if session_id else str(uuid.uuid4()),
            "aggregator": aggregator,
            "aggregator_kwargs": aggregator_kwargs,
            "round_timeout": round_timeout,
            "buffer_size": round_buffer_size,
            "model_id": model_id,
            "rounds": rounds,
            "delete_models_storage": delete_models,
            "clients_required": min_clients,
            "clients_requested": requested_clients,
            "task": (""),
            "validate": validate,
            "helper_type": helper,
        }

        # Start session
        threading.Thread(target=self.control.session, args=(session_config,)).start()

        # Return success response
        return jsonify(
            {
                "success": True,
                "message": "Session started successfully.",
                "config": session_config,
            }
        )
