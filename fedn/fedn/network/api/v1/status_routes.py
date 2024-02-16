
from flask import Blueprint, jsonify, request

from fedn.network.storage.statestore.repositories.status_repository import \
    StatusRepository

from .shared import (api_version, get_post_data_to_kwargs,
                     get_typed_list_headers, get_use_typing, mdb)

bp = Blueprint("status", __name__, url_prefix=f"/api/{api_version}/statuses")

status_repository = StatusRepository(mdb, "control.status")


@bp.route("/", methods=["GET"])
def get_statuses():
    try:
        limit, skip, sort_key, sort_order, use_typing = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        statuses = status_repository.list(limit, skip, sort_key, sort_order, use_typing=use_typing, **kwargs)

        result = [status.__dict__ for status in statuses["result"]] if use_typing else statuses["result"]

        response = {
            "count": statuses["count"],
            "result": result
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/list", methods=["POST"])
def list_statuses():
    try:
        limit, skip, sort_key, sort_order, use_typing = get_typed_list_headers(request.headers)
        kwargs = get_post_data_to_kwargs(request)

        statuses = status_repository.list(limit, skip, sort_key, sort_order, use_typing=use_typing, **kwargs)

        result = [status.__dict__ for status in statuses["result"]] if use_typing else statuses["result"]

        response = {
            "count": statuses["count"],
            "result": result
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/count", methods=["GET", "POST"])
def statuses_count():
    try:
        kwargs = request.args.to_dict() if request.method == "GET" else get_post_data_to_kwargs(request)
        count = status_repository.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@bp.route("/<string:id>", methods=["GET"])
def get_status(id: str):
    try:
        use_typing: bool = get_use_typing(request.headers)
        status = status_repository.get(id, use_typing=use_typing)

        response = status.__dict__ if use_typing else status

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404
