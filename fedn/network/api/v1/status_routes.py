from flask import Blueprint, jsonify, request

from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.v1.shared import api_version, get_post_data_to_kwargs, get_typed_list_headers, get_use_typing, mdb
from fedn.network.storage.statestore.stores.shared import EntityNotFound
from fedn.network.storage.statestore.stores.status_store import StatusStore

bp = Blueprint("status", __name__, url_prefix=f"/api/{api_version}/statuses")

status_store = StatusStore(mdb, "control.status")


@bp.route("/", methods=["GET"])
@jwt_auth_required(role="admin")
def get_statuses():
    """Get statuses
    Retrieves a list of statuses based on the provided parameters.
    By specifying a parameter in the url, you can filter the statuses based on that parameter,
    and the response will contain only the statuses that match the filter.
    ---
    tags:
        - Statuses
    parameters:
      - name: type
        in: query
        required: false
        type: string
        description: Type of the status
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
      - name: session_id
        in: query
        required: false
        type: string
      - name: log_level
        in: query
        required: false
        type: string
        description: Log level of the status
      - name: correlation_id
        in: query
        required: false
        type: string
        description: Correlation id of the status
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of statuses to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of statuses to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the statuses by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the statuses in ('asc' or 'desc')
    definitions:
      Status:
        type: object
        properties:
          id:
            type: string
          status:
            type: string
          session_id:
            type: string
          timestamp:
            type: object
            format: date-time
          type:
            type: string
          data:
            type: string
          correlation_id:
            type: string
          extra:
            type: string
          sender:
            type: object
            properties:
                name:
                    type: string
                role:
                    type: string
          log_level:
            type: object
    responses:
      200:
        description: A list of statuses and the total count.
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
        limit, skip, sort_key, sort_order, use_typing = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        statuses = status_store.list(limit, skip, sort_key, sort_order, use_typing=use_typing, **kwargs)

        result = [status.__dict__ for status in statuses["result"]] if use_typing else statuses["result"]

        response = {"count": statuses["count"], "result": result}

        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/list", methods=["POST"])
@jwt_auth_required(role="admin")
def list_statuses():
    """Get statuses
    Retrieves a list of statuses based on the provided parameters.
    Works much like the GET statuses method, but allows for a more complex query.
    By specifying a parameter in the body, you can filter the statuses based on that parameter,
    and the response will contain only the statuses that match the filter. If the parameter value contains a comma,
    the filter will be an "in" query, meaning that the statuses will be returned if the specified field contains any of the values in the parameter.
    ---
    tags:
        - Statuses
    parameters:
      - name: status
        in: body
        required: false
        schema:
            type: object
            properties:
                type:
                    type: string
                    description: Type of the status
                sender.name:
                    type: string
                    description: Name of the sender
                sender.role:
                    required: false
                    type: string
                    description: Role of the sender
                session_id:
                    required: false
                    type: string
                log_level:
                    required: false
                    type: string
                    description: Log level of the status
                correlation_id:
                    required: false
                    type: string
                    description: Correlation id of the status
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of statuses to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of statuses to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the statuses by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the statuses in ('asc' or 'desc')
    responses:
      200:
        description: A list of statuses and the total count.
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
        limit, skip, sort_key, sort_order, use_typing = get_typed_list_headers(request.headers)
        kwargs = get_post_data_to_kwargs(request)

        statuses = status_store.list(limit, skip, sort_key, sort_order, use_typing=use_typing, **kwargs)

        result = [status.__dict__ for status in statuses["result"]] if use_typing else statuses["result"]

        response = {"count": statuses["count"], "result": result}

        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["GET"])
@jwt_auth_required(role="admin")
def get_statuses_count():
    """Statuses count
    Retrieves the count of statuses based on the provided parameters.
    By specifying a parameter in the url, you can filter the statuses based on that parameter,
    and the response will contain only the count of statuses that match the filter.
    ---
    tags:
        - Statuses
    parameters:
      - name: type
        in: query
        required: false
        type: string
        description: Type of the status
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
      - name: session_id
        in: query
        required: false
        type: string
      - name: log_level
        in: query
        required: false
        type: string
        description: Log level of the status
      - name: correlation_id
        in: query
        required: false
        type: string
        description: Correlation id of the status
    responses:
        200:
            description: The count of statuses
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
        count = status_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["POST"])
@jwt_auth_required(role="admin")
def statuses_count():
    """Statuses count
    Retrieves the count of statuses based on the provided parameters.
    Works much like the GET /statuses/count endpoint, but allows for more complex queries.
    By specifying a parameter in the request body, you can filter the statuses based on that parameter,
    if the parameter value contains a comma, the filter will be an "in" query, meaning that the statuses
    will be returned if the specified field contains any of the values in the parameter.
    ---
    tags:
        - Statuses
    parameters:
      - name: status
        in: body
        required: false
        schema:
            type: object
            properties:
                type:
                    type: string
                    description: Type of the status
                sender.name:
                    type: string
                    description: Name of the sender
                sender.role:
                    required: false
                    type: string
                    description: Role of the sender
                session_id:
                    required: false
                    type: string
                log_level:
                    required: false
                    type: string
                    description: Log level of the status
                correlation_id:
                    required: false
                    type: string
                    description: Correlation id of the status
    responses:
        200:
            description: The count of statuses
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
        count = status_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>", methods=["GET"])
@jwt_auth_required(role="admin")
def get_status(id: str):
    """Get status
    Retrieves a status based on the provided id.
    ---
    tags:
        - Statuses
    parameters:
      - name: id
        in: path
        required: true
        type: string
        description: The id of the status
    responses:
        200:
            description: The status
            schema:
                $ref: '#/definitions/Status'
        404:
            description: The status was not found
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
        use_typing: bool = get_use_typing(request.headers)
        status = status_store.get(id, use_typing=use_typing)

        response = status.__dict__ if use_typing else status

        return jsonify(response), 200
    except EntityNotFound:
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500
