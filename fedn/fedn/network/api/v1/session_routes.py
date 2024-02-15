
from flask import Blueprint, jsonify, request

from fedn.network.storage.statestore.repositories.session_repository import \
    SessionRepository

from .shared import api_version, get_typed_list_headers, mdb

bp = Blueprint("session", __name__, url_prefix=f"/api/{api_version}/sessions")

session_repository = SessionRepository(mdb, "control.sessions")


@bp.route("/", methods=["GET"])
def get_sessions():
    try:
        limit, skip, sort_key, sort_order, _ = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        sessions = session_repository.list(limit, skip, sort_key, sort_order, use_typing=False, **kwargs)

        result = sessions["result"]

        response = {
            "count": sessions["count"],
            "result": result
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/<string:id>", methods=["GET"])
def get_session(id: str):
    try:
        session = session_repository.get(id)
        response = session.__dict__

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404
