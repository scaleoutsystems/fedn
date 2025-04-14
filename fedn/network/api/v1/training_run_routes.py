from flask import Blueprint, jsonify, request

from fedn.common.log_config import logger
from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.shared import training_run_store
from fedn.network.api.v1.shared import api_version, get_typed_list_headers

bp = Blueprint("training_run", __name__, url_prefix=f"/api/{api_version}/training-runs")


@bp.route("/", methods=["GET"])
@jwt_auth_required(role="admin")
def get_training_runs():
    try:
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        analytics = training_run_store.list(limit, skip, sort_key, sort_order, **kwargs)
        count = training_run_store.count(**kwargs)

        response = {"count": count, "result": [analytic.to_dict() for analytic in analytics]}
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500
