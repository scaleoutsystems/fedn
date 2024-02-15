
from flask import Blueprint, jsonify, request

from fedn.network.storage.statestore.repositories.model_repository import \
    ModelRepository

from .shared import api_version, get_limit, get_typed_list_headers, mdb

bp = Blueprint("model", __name__, url_prefix=f"/api/{api_version}/models")

model_repository = ModelRepository(mdb, "control.model")


@bp.route("/", methods=["GET"])
def get_models():
    try:
        limit, skip, sort_key, sort_order, _ = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        models = model_repository.list(limit, skip, sort_key, sort_order, use_typing=False, **kwargs)

        result = models["result"]

        response = {
            "count": models["count"],
            "result": result
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/<string:id>", methods=["GET"])
def get_model(id: str):
    try:
        model = model_repository.get(id, use_typing=False)

        response = model

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@bp.route("/<string:id>/descendants", methods=["GET"])
def get_descendants(id: str):
    try:
        limit = get_limit(request.headers)

        descendants = model_repository.list_descendants(id, limit, use_typing=False)

        response = descendants

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404
