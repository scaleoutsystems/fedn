
from flask import Blueprint, jsonify, request

from fedn.network.storage.statestore.repositories.validation_repository import \
    ValidationRepository

from .shared import (api_version, get_post_data_to_kwargs,
                     get_typed_list_headers, get_use_typing, mdb)

bp = Blueprint("validation", __name__, url_prefix=f"/api/{api_version}/validations")

validation_repository = ValidationRepository(mdb, "control.validations")


@bp.route("/", methods=["GET"])
def get_validations():
    try:
        limit, skip, sort_key, sort_order, use_typing = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        validations = validation_repository.list(limit, skip, sort_key, sort_order, use_typing=use_typing, **kwargs)

        result = [validation.__dict__ for validation in validations["result"]] if use_typing else validations["result"]

        response = {
            "count": validations["count"],
            "result": result
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/list", methods=["POST"])
def list_validations():
    try:
        limit, skip, sort_key, sort_order, use_typing = get_typed_list_headers(request.headers)
        kwargs = get_post_data_to_kwargs(request)

        validations = validation_repository.list(limit, skip, sort_key, sort_order, use_typing=use_typing, **kwargs)

        result = [validation.__dict__ for validation in validations["result"]] if use_typing else validations["result"]

        response = {
            "count": validations["count"],
            "result": result
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/count", methods=["GET", "POST"])
def validations_count():
    try:
        kwargs = request.args.to_dict() if request.method == "GET" else get_post_data_to_kwargs(request)
        count = validation_repository.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@bp.route("/<string:id>", methods=["GET"])
def get_validation(id: str):
    try:
        use_typing: bool = get_use_typing(request.headers)
        validation = validation_repository.get(id, use_typing=use_typing)

        response = validation.__dict__ if use_typing else validation

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404
