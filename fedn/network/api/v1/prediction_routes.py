import threading

from flask import Blueprint, jsonify, request

from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.shared import control
from fedn.network.api.v1.shared import api_version, mdb
from fedn.network.storage.statestore.stores.model_store import ModelStore
from fedn.network.storage.statestore.stores.prediction_store import PredictionStore
from fedn.network.storage.statestore.stores.shared import EntityNotFound

bp = Blueprint("prediction", __name__, url_prefix=f"/api/{api_version}/predict")

prediction_store = PredictionStore(mdb, "control.predictions")
model_store = ModelStore(mdb, "control.model")


@bp.route("/start", methods=["POST"])
@jwt_auth_required(role="admin")
def start_session():
    """Start a new prediction session.
    param: prediction_id: The session id to start.
    type: prediction_id: str
    param: rounds: The number of rounds to run.
    type: rounds: int
    """
    try:
        data = request.json if request.headers["Content-Type"] == "application/json" else request.form.to_dict()
        prediction_id: str = data.get("prediction_id")

        if not prediction_id or prediction_id == "":
            return jsonify({"message": "prediction_id is required"}), 400

        if data.get("model_id") is None:
            count = model_store.count()
            if count == 0:
                return jsonify({"message": "No models available"}), 400
        else:
            try:
                model_id = data.get("model_id")
                _ = model_store.get(model_id)
            except EntityNotFound:
                return jsonify({"message": f"Model {model_id} not found"}), 404

        session_config = {"prediction_id": prediction_id}

        threading.Thread(target=control.prediction_session, kwargs={"config": session_config}).start()

        return jsonify({"message": "Prediction session started"}), 200
    except Exception:
        return jsonify({"message": "Failed to start prediction session"}), 500
