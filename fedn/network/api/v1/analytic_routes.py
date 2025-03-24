from flask import Blueprint, jsonify, request

from fedn.common.log_config import logger
from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.shared import analytic_store
from fedn.network.api.v1.shared import api_version, get_typed_list_headers

bp = Blueprint("analytic", __name__, url_prefix=f"/api/{api_version}/analytics")


@bp.route("/", methods=["GET"])
@jwt_auth_required(role="admin")
def get_analytics():
    try:
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        response = analytic_store.list(limit, skip, sort_key, sort_order, **kwargs)

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/", methods=["POST"])
@jwt_auth_required(role="client")
def add_analytics():
    try:
        data = request.json if request.headers["Content-Type"] == "application/json" else request.form.to_dict()

        successful, result = analytic_store.add(data)
        response = result
        status_code: int = 201 if successful else 400

        return jsonify(response), status_code
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500
