from flask import Blueprint, jsonify, request

from fedn.network.storage.statestore.repositories.package_repository import \
    PackageRepository
from fedn.network.storage.statestore.repositories.shared import EntityNotFound

from .shared import (api_version, get_post_data_to_kwargs,
                     get_typed_list_headers, get_use_typing, mdb)

bp = Blueprint("package", __name__, url_prefix=f"/api/{api_version}/packages")

package_repository = PackageRepository(mdb, "control.package")


@bp.route("/", methods=["GET"])
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
        return jsonify({"message": str(e)}), 500


@bp.route("/list", methods=["POST"])
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
        return jsonify({"message": str(e)}), 500


@bp.route("/count", methods=["GET"])
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
        count = package_repository.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@bp.route("/count", methods=["POST"])
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
        count = package_repository.count(**kwargs)
        response = count
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@bp.route("/<string:id>", methods=["GET"])
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
        use_typing: bool = get_use_typing(request.headers)
        package = package_repository.get(id, use_typing=use_typing)

        response = package.__dict__ if use_typing else package

        return jsonify(response), 200
    except EntityNotFound as e:
        return jsonify({"message": str(e)}), 404
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@bp.route("/active", methods=["GET"])
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
        use_typing: bool = get_use_typing(request.headers)
        package = package_repository.get_active(use_typing=use_typing)
        response = package.__dict__ if use_typing else package

        return jsonify(response), 200
    except EntityNotFound as e:
        return jsonify({"message": str(e)}), 404
    except Exception as e:
        return jsonify({"message": str(e)}), 500
