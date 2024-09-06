from flask import Blueprint, jsonify, request

from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.v1.shared import api_version, client_store, get_post_data_to_kwargs, get_typed_list_headers, mdb
from fedn.network.storage.statestore.stores.combiner_store import CombinerStore
from fedn.network.storage.statestore.stores.shared import EntityNotFound

bp = Blueprint("combiner", __name__, url_prefix=f"/api/{api_version}/combiners")

combiner_store = CombinerStore(mdb, "network.combiners")


@bp.route("/", methods=["GET"])
@jwt_auth_required(role="admin")
def get_combiners():
    """Get combiners
    Retrieves a list of combiners based on the provided parameters.
    By specifying a parameter in the url, you can filter the combiners based on that parameter,
    and the response will contain only the combiners that match the filter.
    ---
    tags:
        - Combiners
    parameters:
      - name: name
        in: query
        required: false
        type: string
        description: The name of the combiner
      - name: address
        in: query
        required: false
        type: string
      - name: ip
        in: query
        required: false
        type: string
        description: The ip of the combiner
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of combiners to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of combiners to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the combiners by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the combiners in ('asc' or 'desc')
    definitions:
      Combiner:
        type: object
        properties:
          id:
            type: string
          name:
            type: string
          address:
            type: string
          config:
            type: object
          fqdn:
            type: string
            description: Fully Qualified Domain Name (FQDN)
          ip:
            type: string
          key:
            type: string
          parent:
            type: object
          port:
            type: integer
          updated_at:
            type: string
            format: date-time
    responses:
      200:
        description: A list of combiners and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Combiner'
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

        combiners = combiner_store.list(limit, skip, sort_key, sort_order, use_typing=False, **kwargs)

        result = combiners["result"]

        response = {"count": combiners["count"], "result": result}

        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/list", methods=["POST"])
@jwt_auth_required(role="admin")
def list_combiners():
    """List combiners
    Retrieves a list of combiners based on the provided parameters.
    Works much like the GET /combiners endpoint, but allows for more complex queries.
    By specifying a parameter in the request body, you can filter the combiners based on that parameter,
    and the response will contain only the combiners that match the filter. If the parameter value contains a comma,
    the filter will be an "in" query, meaning that the combiners will be returned if the specified field contains any of the values in the parameter.
    ---
    tags:
        - Combiners
    parameters:
      - name: combiner
        in: body
        required: false
        type: object
        description: Object containing the parameters to filter the combiners by
        schema:
          type: object
          properties:
            name:
              type: string
            address:
              type: string
            ip:
              type: string
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of combiners to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of combiners to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the combiners by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the combiners in ('asc' or 'desc')
    responses:
      200:
        description: A list of combiners and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Combiner'
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

        combiners = combiner_store.list(limit, skip, sort_key, sort_order, use_typing=False, **kwargs)

        result = combiners["result"]

        response = {"count": combiners["count"], "result": result}

        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["GET"])
@jwt_auth_required(role="admin")
def get_combiners_count():
    """Combiners count
    Retrieves the count of combiners based on the provided parameters.
    By specifying a parameter in the url, you can filter the combiners based on that parameter,
    and the response will contain only the count of combiners that match the filter.
    ---
    tags:
        - Combiners
    parameters:
      - name: name
        in: query
        required: false
        type: string
        description: The name of the combiner
      - name: address
        in: query
        required: false
        type: string
      - name: ip
        in: query
        required: false
        type: string
        description: The ip of the combiner
    responses:
        200:
            description: The count of combiners
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
        count = combiner_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["POST"])
@jwt_auth_required(role="admin")
def combiners_count():
    """Combiners count
    Retrieves the count of combiners based on the provided parameters.
    Works much like the GET /combiners/count endpoint, but allows for more complex queries.
    By specifying a parameter in the request body, you can filter the combiners based on that parameter,
    and the response will contain only the count of combiners that match the filter. If the parameter value contains a comma,
    the filter will be an "in" query, meaning that the combiners will be returned if the specified field contains any of the values in the parameter.
    ---
    tags:
        - Combiners
    parameters:
      - name: combiner
        in: body
        required: false
        type: object
        description: Object containing the parameters to filter the combiners by
        schema:
          type: object
          properties:
            name:
                type: string
            address:
                type: string
            ip:
                type: string
    responses:
        200:
            description: The count of combiners
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
        count = combiner_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>", methods=["GET"])
@jwt_auth_required(role="admin")
def get_combiner(id: str):
    """Get combiner
    Retrieves a combiner based on the provided id.
    ---
    tags:
        - Combiners
    parameters:
      - name: id
        in: path
        required: true
        type: string
        description: The id of the combiner
    responses:
        200:
            description: The combiner
            schema:
                $ref: '#/definitions/Combiner'
        404:
            description: The combiner was not found
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
        combiner = combiner_store.get(id, use_typing=False)
        response = combiner

        return jsonify(response), 200
    except EntityNotFound:
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500

@bp.route("/<string:id>", methods=["DELETE"])
@jwt_auth_required(role="admin")
def delete_combiner(id: str):
    """Delete combiner
    Deletes a combiner based on the provided id.
    ---
    tags:
        - Combiners
    parameters:
      - name: id
        in: path
        required: true
        type: string
        description: The id of the combiner
    responses:
        200:
            description: The combiner was deleted
        404:
            description: The combiner was not found
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
        result: bool = combiner_store.delete(id)
        msg = "Combiner deleted" if result else "Combiner not deleted"

        return jsonify({"message": msg}), 200
    except EntityNotFound:
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/clients/count", methods=["POST"])
@jwt_auth_required(role="admin")
def number_of_clients_connected():
    """Number of clients connected
    Retrieves the number of clients connected to the combiner.
    ---
    tags:
        - Combiners
    parameters:
      - name: combiners
        in: body
        required: true
        type: object
        description: Object containing the ids of the combiners
        schema:
          type: object
          properties:
            combiners:
                type: string
    responses:
        200:
            description: A list of objects containing the number of clients connected to each combiner
            schema:
                type: Array
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
        combiners = data.get("combiners", "")
        combiners = combiners.split(",") if combiners else []
        response = client_store.connected_client_count(combiners)

        result = {
            "result": response
        }

        return jsonify(result), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500
