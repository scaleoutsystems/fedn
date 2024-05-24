import threading

from flask import Blueprint, jsonify, request

from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.shared import control
from fedn.network.api.v1.shared import (api_version, get_post_data_to_kwargs,
                                        get_typed_list_headers, mdb)
from fedn.network.storage.statestore.stores.session_store import SessionStore
from fedn.network.storage.statestore.stores.shared import EntityNotFound

from .model_routes import model_store

bp = Blueprint("inference", __name__, url_prefix=f"/api/{api_version}/infer")


@bp.route("/start", methods=["POST"])
@jwt_auth_required(role="admin")
def start_session():
    """Start a new inference session.
    param: session_id: The session id to start.
    type: session_id: str
    param: rounds: The number of rounds to run.
    type: rounds: int
    """
    try:
        data = request.json if request.headers["Content-Type"] == "application/json" else request.form.to_dict()
        session_id: str = data.get("session_id")

        if not session_id or session_id == "":
            return jsonify({"message": "Session ID is required"}), 400

        session_config = {"session_id": session_id}

        threading.Thread(target=control.inference_session, kwargs={"config":session_config}).start()

        return jsonify({"message": "Session started"}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500
