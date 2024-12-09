import io

import numpy as np
from flask import Blueprint, jsonify, request, send_file

from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.shared import modelstorage_config
from fedn.network.api.v1.shared import api_version, get_limit, get_post_data_to_kwargs, get_reverse, get_typed_list_headers, mdb, minio_repository
from fedn.network.storage.statestore.stores.model_store import ModelStore
from fedn.network.storage.statestore.stores.shared import EntityNotFound

bp = Blueprint("model", __name__, url_prefix=f"/api/{api_version}/models")

model_store = ModelStore(mdb, "control.model")


@bp.route("/", methods=["GET"])
@jwt_auth_required(role="admin")
def get_models():
    """Get models
    Retrieves a list of models based on the provided parameters.
    By specifying a parameter in the url, you can filter the models based on that parameter,
    and the response will contain only the models that match the filter.
    ---
    tags:
      - Models
    parameters:
      - in: query
        name: model
        schema:
        type: string
        description: A unique identifier for the model
      - in: query
        name: parent_model
        schema:
        type: string
        description: The unique identifier of the parent model
      - in: query
        name: session_id
        schema:
        type: string
        description: The unique identifier of the session
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of models to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of models to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the models by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the models in ('asc' or 'desc')
    definitions:
        Model:
            type: object
            properties:
            id:
                type: string
                description: The id of the model
            model:
                type: string
                description: A unique identifier for the model
            parent_model:
                type: string
                description: The unique identifier of the parent model
            session_id:
                type: string
                description: The unique identifier of the session
            committed_at:
                type: string
                format: date-time
                description: The date and time the model was created
    responses:
      200:
        description: A list of models and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Model'
      500:
        description: An error occurred
        schema:
            type: object
            properties:
                error:
                    type: string
    """
    try:
        limit, skip, sort_key, sort_order, _ = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        models = model_store.list(limit, skip, sort_key, sort_order, use_typing=False, **kwargs)

        result = models["result"]

        response = {"count": models["count"], "result": result}

        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/list", methods=["POST"])
@jwt_auth_required(role="admin")
def list_models():
    """List models
    Retrieves a list of models based on the provided parameters.
    Works much like the GET /models endpoint, but allows for more complex queries.
    By specifying a parameter in the body, you can filter the models based on that parameter,
    and the response will contain only the models that match the filter. If the parameter value contains a comma,
    the filter will be an "in" query, meaning that the models will be returned if the specified field contains any of the values in the parameter.
    ---
    tags:
      - Models
    parameters:
      - name: model
        in: body
        required: false
        type: object
        description: Object containing the model filter
        schema:
          type: object
          properties:
            model:
              type: string
              description: A unique identifier for the model
            parent_model:
              type: string
              description: The unique identifier of the parent model
            session_id:
              type: string
              description: The unique identifier of the session
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of models to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of models to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the models by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the models in ('asc' or 'desc')
    responses:
      200:
        description: A list of models and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Model'
      500:
        description: An error occurred
        schema:
            type: object
            properties:
                error:
                    type: string
    """
    try:
        limit, skip, sort_key, sort_order, _ = get_typed_list_headers(request.headers)
        kwargs = get_post_data_to_kwargs(request)

        models = model_store.list(limit, skip, sort_key, sort_order, use_typing=False, **kwargs)

        result = models["result"]

        response = {"count": models["count"], "result": result}

        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["GET"])
@jwt_auth_required(role="admin")
def get_models_count():
    """Models count
    Retrieves the count of models based on the provided parameters.
    By specifying a parameter in the url, you can filter the models based on that parameter,
    and the response will contain only the count of models that match the filter.
    ---
    tags:
        - Models
    parameters:
      - in: query
        name: model
        schema:
        type: string
        description: A unique identifier for the model
      - in: query
        name: parent_model
        schema:
        type: string
        description: The unique identifier of the parent model
      - in: query
        name: session_id
        schema:
        type: string
        description: The unique identifier of the session
    responses:
        200:
            description: The count of models
            schema:
                type: integer
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        kwargs = request.args.to_dict()
        count = model_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["POST"])
@jwt_auth_required(role="admin")
def models_count():
    """Models count
    Retrieves the count of models based on the provided parameters.
    Much like the GET /models/count endpoint, but allows for more complex queries.
    By specifying a parameter in the body, you can filter the models based on that parameter,
    and the response will contain only the count of models that match the filter. If the parameter value contains a comma,
    the filter will be an "in" query, meaning that the models will be returned if the specified field contains any of the values in the parameter.
    ---
    tags:
        - Models
    parameters:
      - name: model
        in: body
        required: false
        type: object
        description: Object containing the model filter
        schema:
          type: object
          properties:
            model:
              type: string
              description: A unique identifier for the model
            parent_model:
              type: string
              description: The unique identifier of the parent model
            session_id:
              type: string
              description: The unique identifier of the session
    responses:
        200:
            description: The count of models
            schema:
                type: integer
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        kwargs = get_post_data_to_kwargs(request)
        count = model_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>", methods=["GET"])
@jwt_auth_required(role="admin")
def get_model(id: str):
    """Get model
    Retrieves a model based on the provided id.
    ---
    tags:
        - Models
    parameters:
      - name: id
        in: path
        required: true
        type: string
        description: The id or model property of the model
    responses:
        200:
            description: The model
            schema:
                $ref: '#/definitions/Model'
        404:
            description: The model was not found
            schema:
                type: object
                properties:
                    message:
                        type: string
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        model = model_store.get(id, use_typing=False)

        response = model

        return jsonify(response), 200
    except EntityNotFound:
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>", methods=["PATCH"])
@jwt_auth_required(role="admin")
def patch_model(id: str):
    """Patch model
    Updates a model based on the provided id. Only the fields that are present in the request will be updated.
    ---
    tags:
        - Models
    parameters:
        - name: id
            in: path
            required: true
            type: string
            description: The id or model property of the model
        - name: model
            in: body
            required: true
            type: object
            description: The model data to update
    responses:
        200:
            description: The updated model
            schema:
                $ref: '#/definitions/Model'
        404:
            description: The model was not found
            schema:
                type: object
                properties:
                    message:
                        type: string
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        model = model_store.get(id, use_typing=False)

        data = request.get_json()
        _id = model["id"]

        # Update the model with the new data
        # Only update the fields that are present in the request
        for key, value in data.items():
            if key in ["_id", "model"]:
                continue
            model[key] = value

        success, message = model_store.update(_id, model)

        if success:
            response = model
            return jsonify(response), 200

        return jsonify({"message": f"Failed to update model: {message}"}), 500
    except EntityNotFound:
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>", methods=["PUT"])
@jwt_auth_required(role="admin")
def put_model(id: str):
    """Put model
    Updates a model based on the provided id. All fields will be updated with the new data.
    ---
    tags:
        - Models
    parameters:
        - name: id
            in: path
            required: true
            type: string
            description: The id or model property of the model
        - name: model
            in: body
            required: true
            type: object
            description: The model data to update
    responses:
        200:
            description: The updated model
            schema:
                $ref: '#/definitions/Model'
        404:
            description: The model was not found
            schema:
                type: object
                properties:
                    message:
                        type: string
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        model = model_store.get(id, use_typing=False)
        data = request.get_json()
        _id = model["id"]

        success, message = model_store.update(_id, data)

        if success:
            response = model
            return jsonify(response), 200

        return jsonify({"message": f"Failed to update model: {message}"}), 500
    except EntityNotFound:
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>/descendants", methods=["GET"])
@jwt_auth_required(role="admin")
def get_descendants(id: str):
    """Get model descendants
    Retrieves a list of model descendants of the provided model id/model property.
    ---
    tags:
        - Models
    parameters:
      - name: id
        in: path
        required: true
        type: string
        description: The id or model property of the model
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of models to retrieve (defaults to 10)
        default: 10
    responses:
        200:
            description: The model
            schema:
                $ref: '#/definitions/Model'
        404:
            description: The inital model was not found
            schema:
                type: object
                properties:
                    message:
                        type: string
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        limit = get_limit(request.headers)

        descendants = model_store.list_descendants(id, limit or 10, use_typing=False)

        response = descendants

        return jsonify(response), 200
    except EntityNotFound:
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>/ancestors", methods=["GET"])
@jwt_auth_required(role="admin")
def get_ancestors(id: str):
    """Get model ancestors
    Retrieves a list of model ancestors of the provided model id/model property.
    ---
    tags:
        - Models
    parameters:
      - name: id
        in: path
        required: true
        type: string
        description: The id or model property of the model
      - name: include_self
        in: query
        required: false
        type: boolean
        description: Whether to include the initial model in the response
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of models to retrieve (defaults to 10)
        default: 10
      - name: X-Reverse
        in: header
        required: false
        type: boolean
        description: Whether to reverse the order of the ancestors
        default: false
    responses:
        200:
            description: The model
            schema:
                $ref: '#/definitions/Model'
        404:
            description: The inital model was not found
            schema:
                type: object
                properties:
                    message:
                        type: string
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        limit = get_limit(request.headers)
        reverse = get_reverse(request.headers)
        include_self_param: str = request.args.get("include_self")

        include_self: bool = include_self_param and include_self_param.lower() == "true"

        ancestors = model_store.list_ancestors(id, limit or 10, include_self=include_self, reverse=reverse, use_typing=False)

        response = ancestors

        return jsonify(response), 200
    except EntityNotFound:
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>/download", methods=["GET"])
@jwt_auth_required(role="admin")
def download(id: str):
    """Download
    Downloads the model file of the provided id.
    ---
    tags:
        - Models
    parameters:
      - name: id
        in: path
        required: true
        type: string
        description: The id or model property of the model
    responses:
        200:
            type: file
            description: The model file
        404:
            description: The model was not found
            schema:
                type: object
                properties:
                    message:
                        type: string
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        if minio_repository is not None:
            model = model_store.get(id, use_typing=False)
            model_id = model["model"]

            file = minio_repository.get_artifact_stream(model_id, modelstorage_config["storage_config"]["storage_bucket"])

            return send_file(file, as_attachment=True, download_name=model_id)
        else:
            return jsonify({"message": "No model storage configured"}), 500
    except EntityNotFound:
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>/parameters", methods=["GET"])
@jwt_auth_required(role="admin")
def get_parameters(id: str):
    """Download
    Downloads parameters of the model of the provided id.
    Please not that this endpoint is only available for models that have been stored as numpy arrays.
    ---
    tags:
        - Models
    parameters:
      - name: id
        in: path
        required: true
        type: string
        description: The id or model property of the model
    responses:
        200:
            description: The model parameters
            schema:
                type: object
                properties:
                    parameters:
                        type: object (array of arrays)
        404:
            description: The model was not found
            schema:
                type: object
                properties:
                    message:
                        type: string
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        if minio_repository is not None:
            model = model_store.get(id, use_typing=False)
            model_id = model["model"]

            file = minio_repository.get_artifact_stream(model_id, modelstorage_config["storage_config"]["storage_bucket"])

            file_bytes = io.BytesIO()
            for chunk in file.stream(32 * 1024):
                file_bytes.write(chunk)
            file_bytes.seek(0)  # Reset the pointer to the beginning of the byte array

            a = np.load(file_bytes)

            weights = []
            for i in range(len(a.files)):
                weights.append(a[str(i)].tolist())

            return jsonify(array=weights), 200
        else:
            return jsonify({"message": "No model storage configured"}), 500
    except EntityNotFound:
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/active", methods=["GET"])
@jwt_auth_required(role="admin")
def get_active_model():
    """Get active model
    Retrieves the active model (id).
    ---
    tags:
        - Models
    responses:
        200:
            description: The active model id
            schema:
                type: string
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        active_model = model_store.get_active()

        response = active_model

        return jsonify(response), 200
    except EntityNotFound:
        return jsonify({"message": "No active model found"}), 404
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/active", methods=["PUT"])
@jwt_auth_required(role="admin")
def set_active_model():
    """Set active model
    Sets the active model (id).
    ---
    tags:
        - Models
    parameters:
      - name: model
        in: body
        required: true
        type: object
        description: The model data to update
    responses:
        200:
            description: The updated active model id
            schema:
                type: string
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        data = request.get_json()
        model_id = data["id"]

        response = model_store.set_active(model_id)

        if response:
            return jsonify({"message": "Active model set"}), 200
        else:
            return jsonify({"message": "Failed to set active model"}), 500
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500
