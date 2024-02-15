
from flask import Blueprint, jsonify, request

from fedn.network.storage.statestore.repositories.round_repository import \
    RoundRepository

from .shared import api_version, get_typed_list_headers, mdb

bp = Blueprint("round", __name__, url_prefix=f"/api/{api_version}/rounds")

round_repository = RoundRepository(mdb, "control.rounds")


@bp.route("/", methods=["GET"])
def get_rounds():
    try:
        limit, skip, sort_key, sort_order, _ = get_typed_list_headers(request.headers)

        kwargs = request.args.to_dict()

        rounds = round_repository.list(limit, skip, sort_key, sort_order, use_typing=False, **kwargs)

        result = rounds["result"]

        response = {
            "count": rounds["count"],
            "result": result
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/<string:id>", methods=["GET"])
def get_round(id: str):
    try:
        round = round_repository.get(id, use_typing=False)
        response = round

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404
