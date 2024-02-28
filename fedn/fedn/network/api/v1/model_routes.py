from flask import Blueprint, jsonify, request

from fedn.network.storage.statestore.stores.model_store import ModelStore
from fedn.network.storage.statestore.stores.shared import EntityNotFound

from fedn.network.api.v1.shared import (api_version, get_limit, get_post_data_to_kwargs,
                     get_typed_list_headers, mdb)

bp = Blueprint("model", __name__, url_prefix=f"/api/{api_version}/models")

model_store = ModelStore(mdb, "control.model")


@bp.route("/", methods=["GET"])
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

        response = {
            "count": models["count"],
            "result": result
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@bp.route("/list", methods=["POST"])
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

        response = {
            "count": models["count"],
            "result": result
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@bp.route("/count", methods=["GET"])
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
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@bp.route("/count", methods=["POST"])
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
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@bp.route("/<string:id>", methods=["GET"])
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
    except EntityNotFound as e:
        return jsonify({"message": str(e)}), 404
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@bp.route("/<string:id>/descendants", methods=["GET"])
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
    except EntityNotFound as e:
        return jsonify({"message": str(e)}), 404
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@bp.route("/<string:id>/ancestors", methods=["GET"])
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

        ancestors = model_store.list_ancestors(id, limit or 10, use_typing=False)

        response = ancestors

        return jsonify(response), 200
    except EntityNotFound as e:
        return jsonify({"message": str(e)}), 404
    except Exception as e:
        return jsonify({"message": str(e)}), 500
