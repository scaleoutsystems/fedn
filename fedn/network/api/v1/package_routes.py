import os
import threading

from flask import Blueprint, jsonify, request, send_from_directory
from werkzeug.security import safe_join

from fedn.common.config import FEDN_COMPUTE_PACKAGE_DIR
from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.shared import control
from fedn.network.api.shared import get_checksum as _get_checksum
from fedn.network.api.shared import package_store, repository
from fedn.network.api.v1.shared import api_version, get_post_data_to_kwargs, get_typed_list_headers
from fedn.network.storage.statestore.stores.shared import EntityNotFound
from fedn.utils.checksum import sha

bp = Blueprint("package", __name__, url_prefix=f"/api/{api_version}/packages")


@bp.route("/", methods=["GET"])
@jwt_auth_required(role="admin")
def get_packages():
    """Get packages
    Retrieves a list of packages based on the provided parameters.
    By specifying a parameter in the url, you can filter the packages based on that parameter,
    and the response will contain only the packages that match the filter.
    ---
    tags:
      - Packages
    parameters:
      - in: query
        name: name
        schema:
        type: string
        description: The name of the package
      - in: query
        name: description
        schema:
        type: string
        description: The description of the package
      - in: query
        name: file_name
        schema:
        type: string
        description: Display name of the file, set by the user
      - in: query
        name: storage_file_name
        schema:
        type: string
        description: Name of the stored file, generated by the system
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of packages to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of packages to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the packages by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the packages in ('asc' or 'desc')
    definitions:
        Package:
            type: object
            properties:
            id:
                type: string
                description: The id of the package
            name:
                type: string
                description: Name of the package
            description:
                type: string
                description: Description of the package
            file_name:
                type: string
                description: Display name of the file
            storage_file_name:
                type: string
                description: Name of the stored file
            key:
                type: string
                description: The key of the type of document. Will be 'package_trail'
            helper:
                type: string
                description: The helper setting for the package
            active:
                type: boolean
                description: Whether the package is active (used if session is started) or not
            committed_at:
                type: string
                format: date-time
                description: The date and time the package was created
    responses:
      200:
        description: A list of packages and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Package'
      500:
        description: An error occurred
        schema:
            type: object
            properties:
                error:
                    type: string

    """
    try:
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = request.args.to_dict()

        response = package_store.list(limit, skip, sort_key, sort_order, **kwargs)

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/list", methods=["POST"])
@jwt_auth_required(role="admin")
def list_packages():
    """List packages
    Retrieves a list of packages based on the provided parameters.
    Works much like the GET /packages endpoint, but allows for more complex queries.
    By specifying a parameter in the body, you can filter the packages based on that parameter,
    and the response will contain only the packages that match the filter. If the parameter value contains a comma,
    the filter will be an "in" query, meaning that the packages will be returned if the specified field contains any of the values in the parameter.
    ---
    tags:
      - Packages
    parameters:
      - name: package
        in: body
        required: false
        type: object
        description: Object containing the package filter
        schema:
          type: object
          properties:
            name:
              type: string
              description: The name of the package
            description:
              type: string
              description: The description of the package
            file_name:
              type: string
              description: The file name of the package, set by the user
            storage_file_name:
              type: string
              description: The name of the stored file, generated by the system
      - name: X-Limit
        in: header
        required: false
        type: integer
        description: The maximum number of packages to retrieve
      - name: X-Skip
        in: header
        required: false
        type: integer
        description: The number of packages to skip
      - name: X-Sort-Key
        in: header
        required: false
        type: string
        description: The key to sort the packages by
      - name: X-Sort-Order
        in: header
        required: false
        type: string
        description: The order to sort the packages in ('asc' or 'desc')
    responses:
      200:
        description: A list of packages and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Package'
      500:
        description: An error occurred
        schema:
            type: object
            properties:
                error:
                    type: string
    """
    try:
        limit, skip, sort_key, sort_order = get_typed_list_headers(request.headers)
        kwargs = get_post_data_to_kwargs(request)

        response = package_store.list(limit, skip, sort_key, sort_order, **kwargs)

        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["GET"])
@jwt_auth_required(role="admin")
def get_packages_count():
    """Package count
    Retrieves the count of packages based on the provided parameters.
    By specifying a parameter in the url, you can filter the packages based on that parameter,
    and the response will contain only the packages that match the filter.
    ---
    tags:
      - Packages
    parameters:
      - in: query
        name: name
        schema:
        type: string
        description: The name of the package
      - in: query
        name: description
        schema:
        type: string
        description: The description of the package
      - in: query
        name: file_name
        schema:
        type: string
        description: Display name of the file, set by the user
      - in: query
        name: storage_file_name
        schema:
        type: string
        description: Name of the stored file, generated by the system
    responses:
      200:
        description: A list of packages and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Package'
      500:
        description: An error occurred
        schema:
            type: object
            properties:
                error:
                    type: string

    """
    try:
        kwargs = request.args.to_dict()
        count = package_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/count", methods=["POST"])
@jwt_auth_required(role="admin")
def packages_count():
    """Package count
    Retrieves the count of packages based on the provided parameters.
    Much like the GET /packages/count endpoint, but allows for more complex queries.
    By specifying a parameter in the body, you can filter the packages based on that parameter,
    and the response will contain only the count of packages that match the filter. If the parameter value contains a comma,
    the filter will be an "in" query, meaning that the packages will be returned if the specified field contains any of the values in the parameter.
    ---
    tags:
      - Packages
    parameters:
      - name: package
        in: body
        required: false
        type: object
        description: Object containing the package filter
        schema:
          type: object
          properties:
            name:
              type: string
              description: The name of the package
            description:
              type: string
              description: The description of the package
            file_name:
              type: string
              description: The file name of the package, set by the user
            storage_file_name:
              type: string
              description: The name of the stored file, generated by the system
    responses:
      200:
        description: A list of packages and the total count.
        schema:
            type: object
            properties:
                count:
                    type: integer
                result:
                    type: array
                    items:
                        $ref: '#/definitions/Package'
      500:
        description: An error occurred
        schema:
            type: object
            properties:
                error:
                    type: string
    """
    try:
        kwargs = get_post_data_to_kwargs(request)
        count = package_store.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/<string:id>", methods=["GET"])
@jwt_auth_required(role="admin")
def get_package(id: str):
    """Get package
    Retrieves a package based on the provided id.
    ---
    tags:
        - Packages
    parameters:
      - name: id
        in: path
        required: true
        type: string
        description: The id or package property of the package
    responses:
        200:
            description: The package
            schema:
                $ref: '#/definitions/package'
        404:
            description: The package was not found
            schema:
                type: object
                properties:
                    message:
                        type: string
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        response = package_store.get(id)

        return jsonify(response), 200
    except EntityNotFound:
        return jsonify({"message": f"Entity with id: {id} not found"}), 404
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/active", methods=["GET"])
@jwt_auth_required(role="admin")
def get_active_package():
    """Get active package
    Retrieves the active package
    ---
    tags:
        - Packages
    responses:
        200:
            description: The package
            schema:
                $ref: '#/definitions/package'
        404:
            description: The package was not found
            schema:
                type: object
                properties:
                    message:
                        type: string
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        response = package_store.get_active()

        return jsonify(response), 200
    except EntityNotFound:
        return jsonify({"message": "Entity not found"}), 404
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/active", methods=["PUT"])
@jwt_auth_required(role="admin")
def set_active_package():
    """Set active package
    Sets the active package
    ---
    tags:
        - Packages
    responses:
        200:
            description: The package was set as active
            schema:
                type: object
                properties:
                    message:
                        type: string
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        data = request.json
        package_id = data["id"]
        response = package_store.set_active(package_id)

        if response:
            return jsonify({"message": "Active package set"}), 200
        else:
            return jsonify({"message": "Active package not set"}), 500
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/active", methods=["DELETE"])
@jwt_auth_required(role="admin")
def delete_active_package():
    """Delete active package
    Deletes the active package
    ---
    tags:
        - Packages
    responses:
        200:
            description: The active package was deleted
            schema:
                type: object
                properties:
                    message:
                        type: string
        404:
            description: There was no active package present
            schema:
                type: object
                properties:
                    message:
                        type: string
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    try:
        package_store.delete_active()
        return jsonify({"message": "Active package deleted"}), 200
    except EntityNotFound:
        return jsonify({"message": "Entity not found"}), 404
    except Exception:
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/", methods=["POST"])
@jwt_auth_required(role="admin")
def upload_package():
    """Upload a package
    Uploads a package to the system. The package is stored in the database and the file is stored in the file system.
    ---
    tags:
      - Packages
    requestBody:
      required: true
      content:
        multipart/form-data:
          schema:
            type: object
            properties:
              name:
                type: string
                description: The name of the package
              description:
                type: string
                description: The description of the package
              file:
                type: string
                format: binary
                description: The package file
              helper:
                type: string
                description: The helper setting for the package
              file_name:
                type: string
                description: The display name of the file
    responses:
      200:
        description: The package was uploaded
        schema:
          type: object
          properties:
            message:
              type: string
      500:
        description: An error occurred
        schema:
          type: object
          properties:
            message:
              type: string
    """
    try:
        data = request.form.to_dict()
        file = request.files["file"]
        file_name = file.filename

        data["file_name"] = file_name

        valid, response = package_store.add(data)

        if not valid:
            return jsonify({"message": response}), 400

        storage_file_name = response["storage_file_name"]
        try:
            file_path = safe_join(FEDN_COMPUTE_PACKAGE_DIR, storage_file_name)
            if not os.path.exists(FEDN_COMPUTE_PACKAGE_DIR):
                os.makedirs(FEDN_COMPUTE_PACKAGE_DIR, exist_ok=True)
            file.save(file_path)
            repository.set_compute_package(storage_file_name, file_path)
        except Exception:
            package_store.delete(response["id"])
            return jsonify({"message": "An unexpected error occurred"}), 500

        package_store.set_active(response["id"])
        return jsonify({"message": "Package uploaded"}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500
        return jsonify({"message": "An unexpected error occurred"}), 500


@bp.route("/download", methods=["GET"])
@jwt_auth_required(role="admin")
def download_package():
    """Download package
    Downloads a package based on the provided id.
    ---
    tags:
        - Packages
    parameters:
      - name: name
        in: query
        required: false
        type: string
        description: The name of the package

    responses:
        200:
            description: The package file
            schema:
                type: object
                properties:
                    message:
                        type: string
        404:
            description: The package was not found
            schema:
                type: object
                properties:
                    message:
                        type: string
        500:
            description: An error occurred
            schema:
                type: object
                properties:
                    message:
                        type: string
    """
    name = request.args.get("name", None)

    if name is None:
        try:
            active_package = package_store.get_active()
            name = active_package["storage_file_name"]
        except EntityNotFound:
            return jsonify({"message": "No active package"}), 404

    try:
        mutex = threading.Lock()
        mutex.acquire()

        return send_from_directory(FEDN_COMPUTE_PACKAGE_DIR, name, as_attachment=True)
    except Exception:
        try:
            data = control.get_compute_package(name)
            # TODO: make configurable, perhaps in config.py or package.py
            file_path = safe_join(FEDN_COMPUTE_PACKAGE_DIR, name)
            with open(file_path, "wb") as fh:
                fh.write(data)
            # TODO: make configurable, perhaps in config.py or package.py
            return send_from_directory(FEDN_COMPUTE_PACKAGE_DIR, name, as_attachment=True)
        except Exception:
            raise
    finally:
        mutex.release()


@bp.route("/checksum", methods=["GET"])
@jwt_auth_required(role="client")
def get_checksum():
    name = request.args.get("name", None)

    success, message, sum = _get_checksum(name)
    if success:
        return jsonify({"message": message, "checksum": sum}), 200
    return jsonify({"message": message, "checksum": sum}), 404
