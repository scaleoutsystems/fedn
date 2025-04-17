from flask import Blueprint, jsonify, request

from fedn.common.log_config import logger
from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.v1.shared import api_version, get_typed_list_headers
from fedn.network.controller.control import Control

bp = Blueprint("training_run", __name__, url_prefix=f"/api/{api_version}/training-runs")


@bp.route("/", methods=["GET"])
@jwt_auth_required(role="admin")
def get_training_runs():
    try:
        db = Control.instance().db
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        training_runs = db.training_run_store.list(limit, skip, sort_key, sort_order, **kwargs)
        count = db.training_run_store.count(**kwargs)

        response = {"count": count, "result": [training_run.to_dict() for training_run in training_runs]}
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500
