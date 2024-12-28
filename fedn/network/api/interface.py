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
            helper_type = package_object["helper"]
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
