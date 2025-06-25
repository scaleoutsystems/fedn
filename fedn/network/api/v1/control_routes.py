from flask import Blueprint, jsonify, request

from fedn.common.log_config import logger
from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.shared import get_db, get_network
from fedn.network.api.v1.session_routes import start_session
from fedn.network.api.v1.shared import api_version, get_post_data_to_kwargs, get_typed_list_headers
from fedn.network.grpc.fedn_pb2 import Command

bp = Blueprint("control", __name__, url_prefix=f"/api/{api_version}/control")


@bp.route("/start_session", methods=["POST"])
@jwt_auth_required(role="admin")
def control_start_session():
    """Start a new session.

    This endpoint is identical to the one in the `session_routes.py` file.
    """
    start_session()


@bp.route("/continue", methods=["POST"])
@jwt_auth_required(role="admin")
def send_continue_signal():
    """Send a continue signal to the controller."""
    network = get_network()

    control = network.get_control()
    control.send_command(Command.CONTINUE)

    return jsonify({"message": "Sent continue signal"}), 200


@bp.route("/stop", methods=["POST"])
@jwt_auth_required(role="admin")
def send_stop_signal():
    network = get_network()
    control = network.get_control()
    control.send_command(Command.STOP)

    return jsonify({"message": "Sent stop signal"}), 200
