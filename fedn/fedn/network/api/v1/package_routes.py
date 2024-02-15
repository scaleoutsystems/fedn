
from flask import Blueprint, jsonify, request

from fedn.network.storage.statestore.repositories.package_repository import \
    PackageRepository

from .shared import api_version, get_typed_list_headers, mdb

bp = Blueprint("package", __name__, url_prefix=f"/api/{api_version}/packages")

package_repository = PackageRepository(mdb, "control.package")


@bp.route("/", methods=["GET"])
def get_packages():
    try:
        limit, skip, sort_key, sort_order, _ = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        packages = package_repository.list(limit, skip, sort_key, sort_order, use_typing=True, **kwargs)

        result = [package.__dict__ for package in packages["result"]]

        response = {
            "count": packages["count"],
            "result": result
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/<string:id>", methods=["GET"])
def get_package(id: str):
    try:
        package = package_repository.get(id, use_typing=True)

        response = package.__dict__

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@bp.route("/active", methods=["GET"])
def get_active_package():
    try:
        package = package_repository.get_active()
        response = package.__dict__

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404
