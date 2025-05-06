from flask import Blueprint, jsonify, request

from fedn.common.log_config import logger
from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.v1.shared import api_version, get_post_data_to_kwargs, get_typed_list_headers
from fedn.network.controller.control import Control
from fedn.network.storage.statestore.stores.dto.attribute import AttributeDTO
from fedn.network.storage.statestore.stores.shared import MissingFieldError, ValidationError

bp = Blueprint("attribute", __name__, url_prefix=f"/api/{api_version}/attributes")


@bp.route("/", methods=["GET"])
@jwt_auth_required(role="admin")
def get_attributes():
    try:
        db = Control.instance().db
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        attributes = db.attribute_store.list(limit, skip, sort_key, sort_order, **kwargs)
        count = db.attribute_store.count(**kwargs)

        response = {"count": count, "result": [attribute.to_dict() for attribute in attributes]}
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/list", methods=["POST"])
@jwt_auth_required(role="admin")
def list_attributes():
    try:
        db = Control.instance().db
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = get_post_data_to_kwargs(request)

        attributes = db.attribute_store.list(limit, skip, sort_key, sort_order, **kwargs)
        count = db.attribute_store.count(**kwargs)

        response = {"count": count, "result": [attribute.to_dict() for attribute in attributes]}
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["GET"])
@jwt_auth_required(role="admin")
def get_attributes_count():
    try:
        db = Control.instance().db
        kwargs = request.args.to_dict()
        count = db.attribute_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["POST"])
@jwt_auth_required(role="admin")
def attributes_count():
    try:
        db = Control.instance().db
        kwargs = request.json if request.headers["Content-Type"] == "application/json" else request.form.to_dict()
        count = db.attribute_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>", methods=["GET"])
@jwt_auth_required(role="admin")
def get_attribute(id: str):
    try:
        db = Control.instance().db
        attribute = db.attribute_store.get(id)
        if attribute is None:
            return jsonify({"message": f"Entity with id: {id} not found"}), 404

        response = attribute.to_dict()
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/", methods=["POST"])
@jwt_auth_required(role="admin")
def add_attributes():
    try:
        db = Control.instance().db
        data = request.json if request.headers["Content-Type"] == "application/json" else request.form.to_dict()

        attribute = AttributeDTO().patch_with(data)
        result = db.attribute_store.add(attribute)
        response = result.to_dict()
        status_code: int = 201

        return jsonify(response), status_code
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


@bp.route("/current", methods=["POST"])
@jwt_auth_required(role="admin")
def get_client_current_attributes():
    """Get current attributes for clients
    ---
    tags:
        - Clients
    parameters:
      - name: client_ids
        in: body
        required: true
        type: array
        items:
          type: string
        description: List of client IDs to retrieve attributes for
    responses:
        200:
            description: A dict of clients and their attributes
            schema:
                type: object
                properties:
                    client_id:
                        type: object
                        additionalProperties:
                            type: string
        400:
            description: Missing required field
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
        json_data = request.get_json()
        client_ids = json_data.get("client_ids")
        if not client_ids:
            return jsonify({"message": "Missing required field: client_ids"}), 400

        response = {}
        for client_id in client_ids:
            client = db.client_store.get(client_id)
            if client is None:
                response[client_id] = f"Entity with client_id: {client_id} not found"
                continue
            attributes = db.attribute_store.get_current_attributes_for_client(client.client_id)
            response[client.client_id] = {}
            for attribute in attributes:
                response[client.client_id][attribute.key] = attribute.value

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500
