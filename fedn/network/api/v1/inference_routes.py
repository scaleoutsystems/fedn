import threading

from flask import Blueprint, jsonify, request

from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.shared import control
from fedn.network.api.v1.shared import api_version, mdb
from fedn.network.storage.statestore.stores.inference_store import InferenceStore

bp = Blueprint("inference", __name__, url_prefix=f"/api/{api_version}/infer")

inference_store = InferenceStore(mdb, "control.inferences")


@bp.route("/start", methods=["POST"])
@jwt_auth_required(role="admin")
def start_session():
    """Start a new inference session.
    param: inference_id: The session id to start.
    type: inference_id: str
    param: rounds: The number of rounds to run.
    type: rounds: int
    """
    try:
        data = request.json if request.headers["Content-Type"] == "application/json" else request.form.to_dict()
        inference_id: str = data.get("inference_id")

        if not inference_id or inference_id == "":
            return jsonify({"message": "Session ID is required"}), 400

        session_config = {"inference_id": inference_id}

        threading.Thread(target=control.inference_session, kwargs={"config": session_config}).start()

        return jsonify({"message": "Inference session started"}), 200
    except Exception:
        return jsonify({"message": "Failed to start inference session"}), 500
