
from flask import Blueprint, jsonify, request

from fedn.network.storage.statestore.repositories.combiner_repository import \
    CombinerRepository

from .shared import api_version, get_typed_list_headers, get_use_typing, mdb

bp = Blueprint("combiner", __name__, url_prefix=f"/api/{api_version}/combiners")

combiner_repository = CombinerRepository(mdb, "network.combiners")


@bp.route("/", methods=["GET"])
def get_combiners():
    try:
        limit, skip, sort_key, sort_order, _ = get_typed_list_headers(request.headers)

        kwargs = request.args.to_dict()

        combiners = combiner_repository.list(limit, skip, sort_key, sort_order, use_typing=False, **kwargs)

        result = combiners["result"]

        response = {
            "count": combiners["count"],
            "result": result
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/<string:id>", methods=["GET"])
def get_combiner(id: str):
    try:
        combiner = combiner_repository.get(id, use_typing=False)
        response = combiner

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404
