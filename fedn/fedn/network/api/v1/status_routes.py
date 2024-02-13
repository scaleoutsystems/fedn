
from flask import Blueprint, jsonify, request

from fedn.network.storage.statestore.repositories.status_repository import \
    StatusRepository

from .shared import api_version, mdb

bp = Blueprint("status", __name__, url_prefix=f"/api/{api_version}/statuses")

status_repository = StatusRepository(mdb, "control.status")


@bp.route("/", methods=["GET"])
def get_statuses():
    try:
        statuses = status_repository.list()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    response = []
    for status in statuses:
        response.append(status.__dict__)

    return jsonify(response), 200


@bp.route("/<string:id>", methods=["GET"])
def get_status(id: str):
    try:
        skip_typing = request.headers.get("Skip-Typing", "false")

        use_typing = False if skip_typing.lower() == "true" else True
        status = status_repository.get(id, use_typing=use_typing)

        result = status.__dict__ if use_typing else status

        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404
