
from flask import Blueprint, jsonify, request

from fedn.network.storage.statestore.repositories.client_repository import \
    ClientRepository

from .shared import (api_version, get_post_data_to_kwargs,
                     get_typed_list_headers, mdb)

bp = Blueprint("client", __name__, url_prefix=f"/api/{api_version}/clients")

client_repository = ClientRepository(mdb, "network.clients")


@bp.route("/", methods=["GET"])
def get_clients():
    """
    Get clients.

    Retrieves a list of clients based on the provided parameters.
    By specifying a parameter in the url, you can filter the clients based on that parameter,
    and the response will contain only the clients that match the filter.

    Url Parameters:
        - name (str)
        - combiner (str)
        - combiner_preferred (str)
        - ip (str)
        - status (str)
        - updated_at (str)

        Example:
        /api/v1/clients?name=client1&combiner=combiner1

    Headers:
        - X-Limit (int): The maximum number of clients to retrieve.
        - X-Skip (int): The number of clients to skip.
        - X-Sort-Key (str): The key to sort the clients by.
        - X-Sort-Order (str): The order to sort the clients in ('asc' or 'desc').

    Returns:
        A JSON response containing the list of clients and the total count.

        Parameters:
        - count (int): The total count of clients.
        - result (list): The list of clients.

        Result parameters:
        - id (str): The ID of the client.
        - name (str): The name of the client.
        - combiner (str): The combiner that the client has connected to.
        - combiner_preferred (bool | str): Combiner name if provided else False.
        - ip (str): The ip of the client.
        - status (str): The status of the client.
        - updated_at (str): The date and time the client was last updated.
        - last_seen (str): The date and time (containing timezone) the client was last seen.

    Raises:
        500 (Internal Server Error): If an exception occurs during the retrieval process.
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
    """
    List clients.

    Retrieves a list of clients based on the provided parameters.
    Works much like the GET /clients endpoint, but allows for more complex queries.
    By specifying a parameter in the request body, you can filter the clients based on that parameter,
    and the response will contain only the clients that match the filter. If the parameter value contains a comma,
    the filter will be an "in" query, meaning that the clients will be returned if the specified field contains any of the values in the parameter.

    Form Data or JSON Input:
        - name (str)
        - combiner (str)
        - combiner_preferred (str)
        - ip (str)
        - status (str)
        - updated_at (str)

        Example:
        {
            "name": "client1,client2",
            "combiner": "combiner1"
        }

    Headers:
        - X-Limit (int): The maximum number of clients to retrieve.
        - X-Skip (int): The number of clients to skip.
        - X-Sort-Key (str): The key to sort the clients by.
        - X-Sort-Order (str): The order to sort the clients in ('asc' or 'desc').

    Returns:
        A JSON response containing the list of clients and the total count.

        Parameters:
        - count (int): The total count of clients.
        - result (list): The list of clients.

        Result parameters:
        - id (str): The ID of the client.
        - name (str): The name of the client.
        - combiner (str): The combiner that the client has connected to.
        - combiner_preferred (bool | str): Combiner name if provided else False.
        - ip (str): The ip of the client.
        - status (str): The status of the client.
        - updated_at (str): The date and time the client was last updated.
        - last_seen (str): The date and time (containing timezone) the client was last seen.

    Raises:
        500 (Internal Server Error): If an exception occurs during the retrieval process.
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


@bp.route("/count", methods=["GET", "POST"])
def clients_count():
    """Example endpoint returning a list of colors by palette
    This is using docstrings for specifications.
    ---
    parameters:
      - name: palette
        in: path
        type: string
        enum: ['all', 'rgb', 'cmyk']
        required: true
        default: all
    definitions:
      Palette:
        type: object
        properties:
          palette_name:
            type: array
            items:
              $ref: '#/definitions/Color'
      Color:
        type: string
    responses:
      200:
        description: A list of colors (may be filtered by palette)
        schema:
          $ref: '#/definitions/Palette'
        examples:
          rgb: ['red', 'green', 'blue']
    """
    try:
        kwargs = request.args.to_dict() if request.method == "GET" else get_post_data_to_kwargs(request)
        count = client_repository.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@bp.route("/<string:id>", methods=["GET"])
def get_client(id: str):
    try:
        client = client_repository.get(id, use_typing=False)

        response = client

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404
