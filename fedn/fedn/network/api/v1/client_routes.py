
from flask import Blueprint, jsonify, request

from fedn.network.storage.statestore.repositories.client_repository import \
    ClientRepository

from .shared import (api_version, get_post_data_to_kwargs,
                     get_typed_list_headers, mdb)

bp = Blueprint("client", __name__, url_prefix=f"/api/{api_version}/clients")

client_repository = ClientRepository(mdb, "network.clients")


@bp.route("/", methods=["GET"])
def get_clients():
    """Get clients
    Retrieves a list of clients based on the provided parameters.
    By specifying a parameter in the url, you can filter the clients based on that parameter,
    and the response will contain only the clients that match the filter.
    ---
    tags:
        - Clients
    parameters:
      - name: name
        in: query
        required: false
        type: string
        description: The name of the client
      - name: combiner
        in: query
        required: false
        type: string
        description: The combiner (id) that the client has connected to
      - name: combiner_preferred
        in: query
        required: false
        type: string
        description: The combiner (id) that the client has preferred to connect to
      - name: ip
        in: query
        required: false
        type: string
        description: The ip of the client
      - name: status
        in: query
        required: false
        type: string
        description: The status of the client
      - name: updated_at
        in: query
        required: false
        type: string
        description: The date and time the client was last updated
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of clients to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of clients to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the clients by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the clients in ('asc' or 'desc')
    definitions:
      Client:
        type: object
        properties:
          name:
            type: string
          combiner:
            type: string
          combiner_preferred:
            type: string
            description: The combiner (id) that the client has preferred to connect or false (boolean) if the client has no preferred combiner
          ip:
            type: string
          status:
            type: string
          updated_at:
            type: string
          last_seen:
            type: string
    responses:
      200:
        description: A list of clients and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Client'
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

        clients = client_repository.list(limit, skip, sort_key, sort_order, use_typing=False, **kwargs)

        result = clients["result"]

        response = {
            "count": clients["count"],
            "result": result
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/list", methods=["POST"])
def list_clients():
    """List clients
    Retrieves a list of clients based on the provided parameters.
    Works much like the GET /clients endpoint, but allows for more complex queries.
    By specifying a parameter in the request body, you can filter the clients based on that parameter,
    and the response will contain only the clients that match the filter. If the parameter value contains a comma,
    the filter will be an "in" query, meaning that the clients will be returned if the specified field contains any of the values in the parameter.
    ---
    tags:
        - Clients
    parameters:
      - name: client
        in: body
        required: false
        type: object
        description: Object containing the parameters to filter the clients by
        schema:
          type: object
          properties:
            name:
              type: string
            combiner:
              type: string
            ip:
              type: string
            status:
              type: string
            updated_at:
              type: string
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of clients to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of clients to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the clients by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the clients in ('asc' or 'desc')
    responses:
      200:
        description: A list of clients and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Client'
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
        clients = client_repository.list(limit, skip, sort_key, sort_order, use_typing=False, **kwargs)

        result = clients["result"]

        response = {
            "count": clients["count"],
            "result": result
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/count", methods=["GET"])
def get_clients_count():
    """Clients count
    Retrieves the total number of clients based on the provided parameters.
    ---
    tags:
        - Clients
    parameters:
      - name: name
        in: query
        required: false
        type: string
        description: The name of the client
      - name: combiner
        in: query
        required: false
        type: string
        description: The combiner (id) that the client has connected to
      - name: combiner_preferred
        in: query
        required: false
        type: string
        description: The combiner (id) that the client has preferred to connect to
      - name: ip
        in: query
        required: false
        type: string
        description: The ip of the client
      - name: status
        in: query
        required: false
        type: string
        description: The status of the client
      - name: updated_at
        in: query
        required: false
        type: string
        description: The date and time the client was last updated
    responses:
      200:
        description: A list of clients and the total count.
        schema:
            type: integer
      404:
        description: The client was not found
        schema:
            type: object
            properties:
                error:
                    type: string
    """
    try:
        kwargs = request.args.to_dict()
        count = client_repository.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@bp.route("/count", methods=["POST"])
def clients_count():
    """Clients count
    Retrieves the total number of clients based on the provided parameters.
    Works much like the GET /clients/count endpoint, but allows for more complex queries.
    By specifying a parameter in the request body, you can filter the clients based on that parameter,
    and the response will contain only the clients that match the filter. If the parameter value contains a comma,
    the filter will be an "in" query, meaning that the clients will be returned if the specified field contains any of the values in the parameter.
    ---
    tags:
        - Clients
    parameters:
      - name: client
        in: body
        required: false
        type: object
        description: Object containing the parameters to filter the clients by
        schema:
          type: object
          properties:
            name:
              type: string
            combiner:
              type: string
            ip:
              type: string
            status:
              type: string
            updated_at:
              type: string
    responses:
      200:
        description: A list of clients and the total count.
        schema:
            type: integer
      404:
        description: The client was not found
        schema:
            type: object
            properties:
                error:
                    type: string
    """
    try:
        kwargs = get_post_data_to_kwargs(request)
        count = client_repository.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@bp.route("/<string:id>", methods=["GET"])
def get_client(id: str):
    """Get client
    Retrieves a client based on the provided id.
    ---
    tags:
        - Clients
    parameters:
      - name: id
        in: path
        required: true
        type: string
        description: The id of the client
    responses:
        200:
            description: A client object
            schema:
                $ref: '#/definitions/Client'
        404:
            description: The client was not found
            schema:
                type: object
                properties:
                    error:
                        type: string
    """
    try:
        client = client_repository.get(id, use_typing=False)

        response = client

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404
