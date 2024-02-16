
from flask import Blueprint, jsonify, request

from fedn.network.storage.statestore.repositories.package_repository import \
    PackageRepository

from .shared import (api_version, get_post_data_to_kwargs,
                     get_typed_list_headers, get_use_typing, mdb)

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


@bp.route("/list", methods=["POST"])
def list_packages():
    try:
        limit, skip, sort_key, sort_order, _ = get_typed_list_headers(request.headers)
        kwargs = get_post_data_to_kwargs(request)

        packages = package_repository.list(limit, skip, sort_key, sort_order, use_typing=True, **kwargs)

        result = [package.__dict__ for package in packages["result"]]

        response = {
            "count": packages["count"],
            "result": result
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/count", methods=["GET", "POST"])
def packages_count():
    try:
        kwargs = request.args.to_dict() if request.method == "GET" else get_post_data_to_kwargs(request)
        count = package_repository.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@bp.route("/<string:id>", methods=["GET"])
def get_package(id: str):
    try:
        use_typing: bool = get_use_typing(request.headers)
        package = package_repository.get(id, use_typing=use_typing)

        response = package.__dict__ if use_typing else package

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@bp.route("/active", methods=["GET"])
def get_active_package():
    try:
        use_typing: bool = get_use_typing(request.headers)
        package = package_repository.get_active(use_typing=use_typing)
        response = package.__dict__ if use_typing else package

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 404
