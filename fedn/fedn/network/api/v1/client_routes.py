
from flask import Blueprint, jsonify, request

from fedn.network.storage.statestore.repositories.client_repository import \
    ClientRepository

from .shared import api_version, get_typed_list_headers, mdb

bp = Blueprint("client", __name__, url_prefix=f"/api/{api_version}/clients")

client_repository = ClientRepository(mdb, "network.clients")


@bp.route("/", methods=["GET"])
def get_clients():
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


@bp.route("/<string:id>", methods=["GET"])
def get_client(id: str):
    try:
        client = client_repository.get(id, use_typing=False)

        response = client

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404
