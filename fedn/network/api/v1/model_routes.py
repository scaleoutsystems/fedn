import io
from io import BytesIO

import numpy as np
from flask import Blueprint, jsonify, request, send_file

from fedn.common.log_config import logger
from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.v1.shared import api_version, get_limit, get_post_data_to_kwargs, get_reverse, get_typed_list_headers
from fedn.network.controller.control import Control
from fedn.network.storage.statestore.stores.shared import EntityNotFound, MissingFieldError, ValidationError

bp = Blueprint("model", __name__, url_prefix=f"/api/{api_version}/models")


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
        db = Control.instance().db
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        models = db.model_store.list(limit, skip, sort_key, sort_order, **kwargs)
        result = [model.to_dict() for model in models]
        count = db.model_store.count(**kwargs)
        response = {"count": count, "result": result}

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
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
        db = Control.instance().db
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = get_post_data_to_kwargs(request)

        models = db.model_store.list(limit, skip, sort_key, sort_order, **kwargs)
        result = [model.to_dict() for model in models]
        count = db.model_store.count(**kwargs)
        response = {"count": count, "result": result}

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
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
        db = Control.instance().db
        kwargs = request.args.to_dict()
        count = db.model_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
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
        db = Control.instance().db
        kwargs = get_post_data_to_kwargs(request)
        count = db.model_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
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
        db = Control.instance().db
        model = db.model_store.get(id)

        if model is None:
            return jsonify({"message": f"Entity with id: {id} not found"}), 404

        response = model.to_dict()
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
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
        db = Control.instance().db
        existing_model = db.model_store.get(id)
        if existing_model is None:
            return jsonify({"message": f"Entity with id: {id} not found"}), 404

        data = request.get_json()

        data.pop("model", None)
        data.pop("model_id", None)

        existing_model.patch_with(data, throw_on_extra_keys=False)
        updated_model = db.model_store.update(existing_model)

        response = updated_model.to_dict()
        return jsonify(response), 200

    except EntityNotFound as e:
        logger.error(f"Entity not found: {e}")
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"message": e.user_message()}), 400
    except MissingFieldError as e:
        logger.error(f"Missing field error: {e}")
        return jsonify({"message": e.user_message()}), 400
    except ValueError as e:
        logger.error(f"ValueError occured: {e}")
        return jsonify({"message": "Invalid object"}), 400
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
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
        db = Control.instance().db
        model = db.model_store.get(id)
        if model is None:
            return jsonify({"message": f"Entity with id: {id} not found"}), 404
        data = request.get_json()
        data.pop("model", None)
        data["model_id"] = id

        model.populate_with(data)
        new_model = db.model_store.update(model)
        response = new_model.to_dict()
        return jsonify(response), 200

    except EntityNotFound as e:
        logger.error(f"Entity not found: {e}")
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"message": e.user_message()}), 400
    except MissingFieldError as e:
        logger.error(f"Missing field error: {e}")
        return jsonify({"message": e.user_message()}), 400
    except ValueError as e:
        logger.error(f"ValueError occured: {e}")
        return jsonify({"message": "Invalid object"}), 400
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
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
        db = Control.instance().db
        limit = get_limit(request.headers)

        descendants = db.model_store.list_descendants(id, limit or 10)

        if descendants is None:
            return jsonify({"message": f"Entity with id: {id} not found"}), 404

        response = [model.to_dict() for model in descendants]
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
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
        db = Control.instance().db
        limit = get_limit(request.headers)
        reverse = get_reverse(request.headers)
        include_self_param: str = request.args.get("include_self")

        include_self: bool = include_self_param and include_self_param.lower() == "true"

        ancestors = db.model_store.list_ancestors(id, limit or 10, include_self=include_self, reverse=reverse)
        if ancestors is None:
            return jsonify({"message": f"Entity with id: {id} not found"}), 404
        response = [model.to_dict() for model in ancestors]
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/leaf-nodes", methods=["GET"])
@jwt_auth_required(role="admin")
def get_leaf_nodes():
    """Get model leaf nodes
    Retrieves a list of
    ---
    tags:
        - Models
    responses:
      200:
        description: A list of models.
        schema:
            type: object
            properties:
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
        db = Control.instance().db
        leaf_nodes = db.model_store.get_leaf_nodes()
        response = [model.to_dict() for model in leaf_nodes]
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
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
        db = Control.instance().db
        repository = Control.instance().repository
        if repository is not None:
            model = db.model_store.get(id)
            if model is None:
                return jsonify({"message": f"Entity with id: {id} not found"}), 404

            file = repository.get_model_stream(model.model_id)

            return send_file(file, as_attachment=True, download_name=model.model_id)
        else:
            return jsonify({"message": "No model storage configured"}), 500
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
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
        db = Control.instance().db
        repository = Control.instance().repository
        if repository is not None:
            model = db.model_store.get(id)
            if model is None:
                return jsonify({"message": f"Entity with id: {id} not found"}), 404

            file = repository.get_model_stream(model.model_id)

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
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/", methods=["POST"])
@jwt_auth_required(role="admin")
def upload_model():
    """Upload model
    Uploads a model to the storage backend.
    ---
    tags:
        - Models
    parameters:
      - name: model
        in: body
        required: true
        type: object
        description: The model data to upload
    responses:
        200:
            description: The uploaded model
            schema:
                $ref: '#/definitions/Model'
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        control = Control.instance()
        data = request.form.to_dict()
        file = request.files["file"]
        name: str = data.get("name", None)

        try:
            object = BytesIO()
            object.seek(0, 0)
            file.seek(0)
            object.write(file.read())
            helper = control.get_helper()
            logger.info(f"Loading model from file using helper {helper.name}")
            object.seek(0)
            model = helper.load(object)
            control.commit(model=model, name=name)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
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

        return jsonify(
            {
                "success": True,
                "message": "Model added successfully",
            }
        ), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


# deprecated


@bp.route("/active", methods=["GET"])
@jwt_auth_required(role="admin")
def get_active_model():
    return jsonify(
        {
            "error": "This endpoint has been deprecated and is no longer available."
            + "The active model concept is no longer used in Fedn. Please use the /models endpoint to retrieve models.",
        }
    ), 410


@bp.route("/active", methods=["PUT"])
@jwt_auth_required(role="admin")
def set_active_model():
    return jsonify(
        {
            "error": "This endpoint has been deprecated and is no longer available. " + "The active model concept is no longer used in Fedn.",
        }
    ), 410
