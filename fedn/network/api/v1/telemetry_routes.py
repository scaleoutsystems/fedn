from flask import Blueprint, jsonify, request

from fedn.common.log_config import logger
from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.v1.shared import api_version, get_typed_list_headers
from fedn.network.controller.control import Control
from fedn.network.storage.statestore.stores.dto.telemetry import TelemetryDTO
from fedn.network.storage.statestore.stores.shared import MissingFieldError, ValidationError

bp = Blueprint("telemetry", __name__, url_prefix=f"/api/{api_version}/telemetry")


@bp.route("/", methods=["GET"])
@jwt_auth_required(role="admin")
def get_telemetries():
    try:
        db = Control.instance().db
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        telemetries = db.telemetry_store.list(limit, skip, sort_key, sort_order, **kwargs)
        count = db.telemetry_store.count(**kwargs)

        response = {"count": count, "result": [telemetry.to_dict() for telemetry in telemetries]}
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/", methods=["POST"])
@jwt_auth_required(role="admin")
def add_telemetries():
    try:
        db = Control.instance().db
        data = request.json if request.headers["Content-Type"] == "application/json" else request.form.to_dict()

        telemetry = TelemetryDTO().patch_with(data)
        result = db.telemetry_store.add(telemetry)
        response = result.to_dict()
        status_code: int = 201

        return jsonify(response), status_code
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"message": e.user_message()}), 400
    except MissingFieldError as e:
        logger.error(f"Missing field error: {e}")
        return jsonify({"message": e.user_message()}), 400
    except ValueError as e:
        logger.error(f"ValueError occured: {e}")
        return jsonify({"message": "Invalid object"}), 400
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred"}), 500
