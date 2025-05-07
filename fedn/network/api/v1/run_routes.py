from flask import Blueprint, jsonify, request

from fedn.common.log_config import logger
from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.v1.shared import api_version, get_post_data_to_kwargs, get_typed_list_headers
from fedn.network.controller.control import Control

bp = Blueprint("run", __name__, url_prefix=f"/api/{api_version}/runs")


@bp.route("/", methods=["GET"])
@jwt_auth_required(role="admin")
def get_runs():
    try:
        db = Control.instance().db
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        runs = db.run_store.list(limit, skip, sort_key, sort_order, **kwargs)
        count = db.run_store.count(**kwargs)

        response = {"count": count, "result": [run.to_dict() for run in runs]}
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/list", methods=["POST"])
@jwt_auth_required(role="admin")
def list_runs():
    try:
        db = Control.instance().db
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = get_post_data_to_kwargs(request)

        result = db.run_store.list(limit, skip, sort_key, sort_order, **kwargs)
        count = db.run_store.count(**kwargs)
        response = {"count": count, "result": [run.to_dict() for run in result]}

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["GET"])
@jwt_auth_required(role="admin")
def get_runs_count():
    try:
        db = Control.instance().db
        kwargs = request.args.to_dict()
        count = db.run_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["POST"])
@jwt_auth_required(role="admin")
def runs_count():
    try:
        db = Control.instance().db
        kwargs = get_post_data_to_kwargs(request)
        count = db.run_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>", methods=["GET"])
@jwt_auth_required(role="admin")
def get_run(id: str):
    try:
        db = Control.instance().db
        response = db.run_store.get(id)
        if response is None:
            return jsonify({"message": f"Entity with id: {id} not found"}), 404
        return jsonify(response.to_dict()), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500
