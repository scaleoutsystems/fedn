from datetime import datetime

from flask import Blueprint, jsonify, request

from fedn.common.config import get_controller_config, get_network_config
from fedn.common.log_config import logger
from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.shared import get_checksum
from fedn.network.api.v1.shared import api_version, get_post_data_to_kwargs, get_typed_list_headers
from fedn.network.controller.control import Control
from fedn.network.storage.statestore.stores.dto import ClientDTO
from fedn.network.storage.statestore.stores.shared import MissingFieldError, ValidationError

bp = Blueprint("client", __name__, url_prefix=f"/api/{api_version}/clients")


@bp.route("/", methods=["GET"])
@jwt_auth_required(role="admin")
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
                message:
                    type: string
    """
    try:
        db = Control.instance().db
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        clients = db.client_store.list(limit, skip, sort_key, sort_order, **kwargs)
        count = db.client_store.count(**kwargs)
        response = {"count": count, "result": [client.to_dict() for client in clients]}

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/list", methods=["POST"])
@jwt_auth_required(role="admin")
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
                message:
                    type: string
    """
    try:
        db = Control.instance().db
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = get_post_data_to_kwargs(request)

        clients = db.client_store.list(limit, skip, sort_key, sort_order, **kwargs)
        count = db.client_store.count(**kwargs)
        response = {"count": count, "result": [client.to_dict() for client in clients]}

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["GET"])
@jwt_auth_required(role="admin")
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
        count = db.client_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["POST"])
@jwt_auth_required(role="admin")
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
        count = db.client_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>", methods=["GET"])
@jwt_auth_required(role="admin")
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
        client = db.client_store.get(id)
        if client is None:
            return jsonify({"message": f"Entity with id: {id} not found"}), 404
        response = client.to_dict()
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>", methods=["DELETE"])
@jwt_auth_required(role="admin")
def delete_client(id: str):
    """Delete client
    Deletes a client based on the provided id.
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
            description: The client was deleted
        404:
            description: The client was not found
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
        result: bool = db.client_store.delete(id)
        if result is False:
            return jsonify({"message": f"Entity with id: {id} not found"}), 404

        msg = "Client deleted" if result else "Client not deleted"

        return jsonify({"message": msg}), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/add", methods=["POST"])
@jwt_auth_required(role="client")
def add_client():
    """Add client
    Adds a client to the network.
    ---
    tags:
        - Clients
    parameters:
      - name: client
        in: body
        required: true
        type: object
        description: Object containing the parameters to create the client
        schema:
          type: object
          properties:
            name:
              type: string
            combiner:
              type: string
            combiner_preferred:
              type: string
            ip:
              type: string
            status:
              type: string
    responses:
        200:
            description: The client was added
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
        network = Control.instance().network
        json_data = request.get_json()
        remote_addr = request.remote_addr

        client_id = json_data.get("client_id", None)
        name = json_data.get("name", None)
        preferred_combiner = json_data.get("combiner_preferred", None)
        package = json_data.get("package", "local")
        helper_type: str = ""

        if package == "remote":
            package_object = db.package_store.get_active()
            if package_object is None:
                return jsonify(
                    {
                        "success": False,
                        "status": "retry",
                        "message": "No compute package found. Set package in controller.",
                    }
                ), 203
            helper_type = package_object.helper
        else:
            helper_type = ""

        if preferred_combiner:
            combiner = network.get_combiner(preferred_combiner)
            if combiner is None:
                return jsonify(
                    {
                        "success": False,
                        "message": f"Combiner {preferred_combiner} not found or unavailable.",
                    },
                    400,
                )
        else:
            combiner = network.find_available_combiner()
            if combiner is None:
                return jsonify({"success": False, "message": "No combiner available."}), 400

        if db.client_store.get(client_id) is None:
            logger.info("Adding client {}".format(client_id))

            last_seen = datetime.now()

            new_client = ClientDTO(
                client_id=client_id,
                name=name,
                combiner=combiner.name,
                combiner_preferred=preferred_combiner,
                ip=remote_addr,
                status="available",
                package=package,
                last_seen=last_seen,
            )

            added_client = db.client_store.add(new_client)
            client_id = added_client.client_id

        payload = {
            "status": "assigned",
            "host": combiner.address,
            "fqdn": combiner.fqdn,
            "package": package,
            "ip": combiner.ip,
            "port": combiner.port,
            "helper_type": helper_type,
            "client_id": client_id,
        }
        return jsonify(payload), 200
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
        return jsonify({"success": False, "message": "An unexpected error occurred"}), 500


@bp.route("/config", methods=["GET"])
@jwt_auth_required(role="admin")
def get_client_config():
    """Get client config
    Retrieves the client configuration.
    ---
    tags:
        - Clients
    responses:
        200:
            description: The client configuration
            schema:
                type: object
                properties:
                    network_id:
                        type: string
                    discover_host:
                        type: string
                    discover_port:
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
        checksum_arg = request.args.get("checksum", "true")
        include_checksum = checksum_arg.lower() == "true"

        config = get_controller_config()
        network_id = get_network_config()
        port = config["port"]
        host = config["host"]
        payload = {
            "network_id": network_id,
            "discover_host": host,
            "discover_port": port,
        }

        if include_checksum:
            success, _, checksum_str = get_checksum()
            if success:
                payload["checksum"] = checksum_str

        return jsonify(payload), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>/attributes", methods=["GET"])
@jwt_auth_required(role="admin")
def get_client_attributes(id):
    """Get client attributes
    Retrieves the attributes of a client based on the provided id.
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
            description: A list of attributes for the client
            schema:
                type: array
                items:
                    type: object
                    properties:
                        key:
                            type: string
                        value:
                            type: string
        404:
            description: The client was not found
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

        client = db.client_store.get(id)
        if client is None:
            return jsonify({"message": f"Entity with id: {id} not found"}), 404

        attributes = db.attribute_store.get_current_attributes_for_client(client.client_id)
        response = {}
        for attribute in attributes:
            response[attribute.key] = attribute.value
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500
