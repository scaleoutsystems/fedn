from flask import Blueprint, jsonify, request

from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.v1.shared import api_version, get_post_data_to_kwargs, get_typed_list_headers, mdb
from fedn.network.storage.statestore.stores.round_store import RoundStore
from fedn.network.storage.statestore.stores.shared import EntityNotFound

bp = Blueprint("round", __name__, url_prefix=f"/api/{api_version}/rounds")

round_store = RoundStore(mdb, "control.rounds")


@bp.route("/", methods=["GET"])
@jwt_auth_required(role="admin")
def get_rounds():
    """Get rounds
    Retrieves a list of rounds based on the provided parameters.
    By specifying a parameter in the url, you can filter the rounds based on that parameter,
    and the response will contain only the rounds that match the filter.
    ---
    tags:
        - Rounds
    parameters:
      - name: status
        in: query
        required: false
        type: string
        description: Status of the round
      - name: round_id
        in: query
        required: false
        type: string
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of rounds to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of rounds to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the rounds by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the rounds in ('asc' or 'desc')
    definitions:
      Round:
        type: object
        properties:
          id:
            type: string
          status:
            type: string
          round_id:
            type: string
          round_config:
            type: object
          round_data:
            type: object
          combiners:
            type: array
            description: List of combiner objects used for the round
            items:
              type: object
    responses:
      200:
        description: A list of rounds and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Round'
      500:
        description: An error occurred
        schema:
            type: object
            properties:
                message:
                    type: string
    """
    try:
        limit, skip, sort_key, sort_order, _ = get_typed_list_headers(request.headers)

        kwargs = request.args.to_dict()

        rounds = round_store.list(limit, skip, sort_key, sort_order, use_typing=False, **kwargs)

        result = rounds["result"]

        response = {"count": rounds["count"], "result": result}

        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/list", methods=["POST"])
@jwt_auth_required(role="admin")
def list_rounds():
    """List rounds
    Retrieves a list of rounds based on the provided parameters.
    Works much like the GET /rounds endpoint, but allows for more complex queries.
    By specifying a parameter in the request body, you can filter the rounds based on that parameter,
    and the response will contain only the rounds that match the filter. If the parameter value contains a comma,
    the filter will be an "in" query, meaning that the rounds will be returned if the specified field contains any of the values in the parameter.
    ---
    tags:
      - Rounds
    parameters:
      - name: round
        in: body
        required: false
        schema:
          type: object
          properties:
            status:
              type: string
            round_id:
              type: string
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of rounds to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of rounds to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the rounds by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the rounds in ('asc' or 'desc')
    responses:
      200:
        description: A list of rounds and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Round'
      500:
        description: An error occurred
        schema:
            type: object
            properties:
                message:
                    type: string
    """
    try:
        limit, skip, sort_key, sort_order, _ = get_typed_list_headers(request.headers)

        kwargs = get_post_data_to_kwargs(request)

        rounds = round_store.list(limit, skip, sort_key, sort_order, use_typing=False, **kwargs)

        result = rounds["result"]

        response = {"count": rounds["count"], "result": result}

        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["GET"])
@jwt_auth_required(role="admin")
def get_rounds_count():
    """Rounds count
    Retrieves the count of rounds based on the provided parameters.
    By specifying a parameter in the url, you can filter the rounds based on that parameter,
    and the response will contain only the count of rounds that match the filter.
    ---
    tags:
        - Rounds
    parameters:
      - name: round_id
        in: query
        required: false
        type: string
      - name: status
        in: query
        required: false
        type: string
    responses:
        200:
            description: The count of rounds
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
        count = round_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["POST"])
@jwt_auth_required(role="admin")
def rounds_count():
    """Rounds count
    Retrieves the count of rounds based on the provided parameters.
    Works much like the GET /rounds/count endpoint, but allows for more complex queries.
    By specifying a parameter in the request body, you can filter the rounds based on that parameter,
    if the parameter value contains a comma, the filter will be an "in" query, meaning that the rounds
    will be returned if the specified field contains any of the values in the parameter.
    ---
    tags:
        - Rounds
    parameters:
      - name: round
        in: body
        required: false
        schema:
          type: object
          properties:
            status:
              type: string
            round_id:
              type: string
    responses:
        200:
            description: The count of rounds
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
        count = round_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>", methods=["GET"])
@jwt_auth_required(role="admin")
def get_round(id: str):
    """Get round
    Retrieves a round based on the provided id.
    ---
    tags:
        - Rounds
    parameters:
      - name: id
        in: path
        required: true
        type: string
        description: The id of the round
    responses:
        200:
            description: The round
            schema:
                $ref: '#/definitions/Round'
        404:
            description: The round was not found
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
        round = round_store.get(id, use_typing=False)
        response = round

        return jsonify(response), 200
    except EntityNotFound:
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500
