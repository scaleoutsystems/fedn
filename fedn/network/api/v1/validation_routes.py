from flask import Blueprint, jsonify, request

from fedn.common.log_config import logger
from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.v1.shared import api_version, get_post_data_to_kwargs, get_typed_list_headers
from fedn.network.controller.control import Control

bp = Blueprint("validation", __name__, url_prefix=f"/api/{api_version}/validations")


@bp.route("/", methods=["GET"])
@jwt_auth_required(role="admin")
def get_validations():
    """Get validations
    Retrieves a list of validations based on the provided parameters.
    By specifying a parameter in the url, you can filter the validations based on that parameter,
    and the response will contain only the validations that match the filter.
    ---
    tags:
        - Validations
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
      - name: session_id
        in: query
        required: false
        type: string
      - name: model_id
        in: query
        required: false
        type: string
      - name: correlation_id
        in: query
        required: false
        type: string
        description: Correlation id of the validation
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of validations to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of validations to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the validations by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the validations in ('asc' or 'desc')
    definitions:
      Validation:
        type: object
        properties:
          id:
            type: string
          correlation_id:
            type: string
          session_id:
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
        description: A list of validations and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Session'
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
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        result = db.validation_store.list(limit, skip, sort_key, sort_order, **kwargs)
        count = db.validation_store.count(**kwargs)
        response = {"count": count, "result": [validation.to_dict() for validation in result]}

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/list", methods=["POST"])
@jwt_auth_required(role="admin")
def list_validations():
    """Get validations
    Retrieves a list of validations based on the provided parameters.
    Works much like the GET validations method, but allows for a more complex query.
    By specifying a parameter in the body, you can filter the validations based on that parameter,
    and the response will contain only the validations that match the filter. If the parameter value contains a comma,
    the filter will be an "in" query, meaning that the validations will be returned if the specified field contains any of the values in the parameter.
    ---
    tags:
        - Validations
    parameters:
      - name: validation
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
                session_id:
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
        description: The maximum number of validations to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of validations to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the validations by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the validations in ('asc' or 'desc')
    responses:
      200:
        description: A list of validations and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Session'
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
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = get_post_data_to_kwargs(request)

        result = db.validation_store.list(limit, skip, sort_key, sort_order, **kwargs)
        count = db.validation_store.count(**kwargs)
        response = {"count": count, "result": [validation.to_dict() for validation in result]}

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["GET"])
@jwt_auth_required(role="admin")
def get_validations_count():
    """Validations count
    Retrieves the count of validations based on the provided parameters.
    By specifying a parameter in the url, you can filter the validations based on that parameter,
    and the response will contain only the count of validations that match the filter.
    ---
    tags:
        - Validations
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
      - name: session_id
        in: query
        required: false
        type: string
      - name: model_id
        in: query
        required: false
        type: string
      - name: correlation_id
        in: query
        required: false
        type: string
        description: Correlation id of the validation
    responses:
        200:
            description: The count of validations
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
        count = db.validation_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["POST"])
@jwt_auth_required(role="admin")
def validations_count():
    """Validations count
    Retrieves the count of validations based on the provided parameters.
    Works much like the GET /validations/count endpoint, but allows for more complex queries.
    By specifying a parameter in the request body, you can filter the validations based on that parameter,
    if the parameter value contains a comma, the filter will be an "in" query, meaning that the validations
    will be returned if the specified field contains any of the values in the parameter.
    ---
    tags:
        - Validations
    parameters:
      - name: validation
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
                session_id:
                    required: false
                    type: string
                model_id:
                    required: false
                    type: string
                correlation_id:
                    required: false
                    type: string
                    description: Correlation id of the status
    responses:
        200:
            description: The count of validations
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
        count = db.validation_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>", methods=["GET"])
@jwt_auth_required(role="admin")
def get_validation(id: str):
    """Get validation
    Retrieves a validation based on the provided id.
    ---
    tags:
        - Validations
    parameters:
      - name: id
        in: path
        required: true
        type: string
        description: The id of the validation
    responses:
        200:
            description: The validation
            schema:
                $ref: '#/definitions/Validation'
        404:
            description: The validation was not found
            schema:
                type: object
                properties:
                    error:
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
        response = db.validation_store.get(id)
        if response is None:
            return jsonify({"message": f"Entity with id: {id} not found"}), 404
        return jsonify(response.to_dict()), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500
