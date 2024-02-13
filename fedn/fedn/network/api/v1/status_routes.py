
from flask import Blueprint, jsonify

from .shared import api_version

statuses = [
    {"id": 1, "status": "ok"},
    {"id": 2, "status": "not ok"},
]


bp = Blueprint("status", __name__, url_prefix=f"/api/{api_version}/statuses")


@bp.route("/", methods=["GET"])
def get_statuses():
    return jsonify(statuses), 200


@bp.route("/<int:status_id>", methods=["GET"])
def get_status(status_id):
    status = [status for status in statuses if status["id"] == status_id]
    return jsonify(status), 200
