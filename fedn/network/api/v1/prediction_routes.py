import threading

from flask import Blueprint, jsonify, request

from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.shared import control
from fedn.network.api.v1.shared import api_version, mdb, get_typed_list_headers, get_post_data_to_kwargs
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


@bp.route("/", methods=["GET"])
@jwt_auth_required(role="admin")
def get_predictions():
    """Get predictions
    Retrieves a list of predictions based on the provided parameters.
    By specifying a parameter in the url, you can filter the predictions based on that parameter,
    and the response will contain only the predictions that match the filter.
    ---
    tags:
        - Predictions
    parameters:
      - name: sender.name
        in: query
        required: false
        type: string
        description: Name of the sender
      - name: sender.role
        in: query
        required: false
        type: string
        description: Role of the sender
      - name: receiver.name
        in: query
        required: false
        type: string
        description: Name of the receiver
      - name: receiver.role
        in: query
        required: false
        type: string
        description: Role of the receiver
      - name: prediction_id
        in: query
        required: false
        type: string
        description: Prediction id of the prediction
      - name: model_id
        in: query
        required: false
        type: string
      - name: correlation_id
        in: query
        required: false
        type: string
        description: Correlation id of the prediction
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of predictions to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of predictions to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the predictions by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the predictions in ('asc' or 'desc')
    definitions:
      Prediction:
        type: object
        properties:
          id:
            type: string
          correlation_id:
            type: string
          prediction_id:
            type: string
          model_id:
            type: string
          timestamp:
            type: object
            format: date-time
          data:
            type: string
          meta:
            type: string
          sender:
            type: object
            properties:
                name:
                    type: string
                role:
                    type: string
          receiver:
            type: object
            properties:
                name:
                    type: string
                role:
                    type: string
    responses:
      200:
        description: A list of predictions and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Prediction'
      500:
        description: An error occurred
        schema:
            type: object
            properties:
                message:
                    type: string
    """
    try:
        limit, skip, sort_key, sort_order, use_typing = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        predictions = prediction_store.list(limit, skip, sort_key, sort_order, use_typing=use_typing, **kwargs)

        result = [prediction.__dict__ for prediction in predictions["result"]] if use_typing else predictions["result"]

        response = {"count": predictions["count"], "result": result}

        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/list", methods=["POST"])
@jwt_auth_required(role="admin")
def list_predictions():
    """List predictions
    Retrieves a list of predictions based on the provided parameters.
    Works much like the GET predictions method, but allows for a more complex query.
    By specifying a parameter in the body, you can filter the predictions based on that parameter,
    and the response will contain only the predictions that match the filter. If the parameter value contains a comma,
    the filter will be an "in" query, meaning that the predictions will be returned if the specified field contains any of the values in the parameter.
    ---
    tags:
        - Predictions
    parameters:
      - name: prediction
        in: body
        required: false
        schema:
            type: object
            properties:
                sender.name:
                    type: string
                    description: Name of the sender
                sender.role:
                    required: false
                    type: string
                    description: Role of the sender
                receiver.name:
                    type: string
                    description: Name of the receiver
                receiver.role:
                    required: false
                    type: string
                    description: Role of the receiver
                prediction_id:
                    required: false
                    type: string
                model_id:
                    required: false
                    type: string
                correlation_id:
                    required: false
                    type: string
                    description: Correlation id of the status
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of predictions to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of predictions to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the predictions by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the predictions in ('asc' or 'desc')
    responses:
      200:
        description: A list of predictions and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Prediction'
      500:
        description: An error occurred
        schema:
            type: object
            properties:
                message:
                    type: string
    """
    try:
        limit, skip, sort_key, sort_order, use_typing = get_typed_list_headers(request.headers)
        kwargs = get_post_data_to_kwargs(request)

        predictions = prediction_store.list(limit, skip, sort_key, sort_order, use_typing=use_typing, **kwargs)

        result = [prediction.__dict__ for prediction in predictions["result"]] if use_typing else predictions["result"]

        response = {"count": predictions["count"], "result": result}

        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500
