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
