
from flask import Blueprint, jsonify, request

from fedn.network.storage.statestore.repositories.model_repository import \
    ModelRepository

from .shared import api_version, get_typed_list_headers, get_use_typing, mdb

bp = Blueprint("model", __name__, url_prefix=f"/api/{api_version}/models")

model_repository = ModelRepository(mdb, "control.model")


@bp.route("/", methods=["GET"])
def get_models():
    try:
        limit, skip, sort_key, sort_order, use_typing = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        models = model_repository.list(limit, skip, sort_key, sort_order, use_typing=use_typing, **kwargs)

        result = [model.__dict__ for model in models["result"]] if use_typing else models["result"]

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
        use_typing: bool = get_use_typing(request.headers)
        model = model_repository.get(id, use_typing=use_typing)

        response = model.__dict__ if use_typing else model

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404
