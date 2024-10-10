from flask import Blueprint, jsonify, request

from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.v1.shared import api_version, package_store
from fedn.network.storage.statestore.stores.shared import EntityNotFound

bp = Blueprint("helper", __name__, url_prefix=f"/api/{api_version}/helpers")



@bp.route("/active", methods=["GET"])
@jwt_auth_required(role="admin")
def get_active_helper():
    """Get active helper
    Retrieves the active helper
    ---
    tags:
        - Helpers
    responses:
        200:
            description: Active helper
        404:
            description: No active helper
        500:
            description: An unexpected error occurred
    """
    try:

        active_package = package_store.get_active()

        response = active_package["helper"]

        return jsonify(response), 200
    except EntityNotFound:
        return jsonify({"message": "No active helper"}), 404
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500

@bp.route("/active", methods=["PUT"])
@jwt_auth_required(role="admin")
def set_active_helper():
    """Set active helper
    Sets the active helper
    ---
    tags:
        - Helpers
    responses:
        200:
            description: Active helper set
        500:
            description: An unexpected error occurred
    """
    try:
        data = request.get_json()
        helper = data["helper"]
        package_store.set_active_helper(helper)

        return jsonify({"message": "Active helper set"}), 200
    except ValueError:
        return jsonify({"message": "Helper is required to be either 'numpyhelper', 'binaryhelper' or 'androidhelper'"}), 400
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500
