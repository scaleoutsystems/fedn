import os

from flask import Flask, jsonify, request

from fedn.common.config import get_controller_config
from fedn.network.api import gunicorn_app
from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.interface import API
from fedn.network.api.shared import control, statestore
from fedn.network.api.v1 import _routes
from fedn.network.api.v1.graphql.schema import schema
from fedn.network.state import ReducerState, ReducerStateToString

custom_url_prefix = os.environ.get("FEDN_CUSTOM_URL_PREFIX", False)
api = API(statestore, control)
app = Flask(__name__)
for bp in _routes:
    app.register_blueprint(bp)
    if custom_url_prefix:
        app.register_blueprint(bp, name=f"{bp.name}_custom", url_prefix=f"{custom_url_prefix}{bp.url_prefix}")


@app.route("/health", methods=["GET"])
def health_check():
    return "OK", 200


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/health", view_func=health_check, methods=["GET"])


@app.route("/api/v1/graphql", methods=["POST"])
def graphql_endpoint():
    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Missing query in request"}), 400

    # Execute the GraphQL query
    result = schema.execute(
        data["query"],
        variables=data.get("variables"),
        context_value={"request": request},  # Pass Flask request object as context if needed
    )

    # Format the result as a JSON response
    response = {"data": result.data}
    if result.errors:
        response["errors"] = [str(error) for error in result.errors]

    return jsonify(response)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/api/v1/graphql", view_func=graphql_endpoint, methods=["POST"])


@app.route("/get_controller_status", methods=["GET"])
@jwt_auth_required(role="admin")
def get_controller_status():
    """Get the status of the controller.
    return: The status as a json object.
    rtype: json
    """
    return jsonify({"state": ReducerStateToString(control.state())}), 200


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_controller_status", view_func=get_controller_status, methods=["GET"])


@app.route("/get_client_config", methods=["GET"])
@jwt_auth_required(role="admin")
def get_client_config():
    """Get the client configuration.
    return: The client configuration as a json object.
    rtype: json
    """
    checksum_arg = request.args.get("checksum", "true")
    checksum = checksum_arg.lower() != "false"
    return api.get_client_config(checksum)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_client_config", view_func=get_client_config, methods=["GET"])


@app.route("/add_combiner", methods=["POST"])
@jwt_auth_required(role="combiner")
def add_combiner():
    """Add a combiner to the network.
    return: The response from the statestore.
    rtype: json
    """
    json_data = request.get_json()
    remote_addr = request.remote_addr
    try:
        response = api.add_combiner(**json_data, remote_addr=remote_addr)
    except TypeError:
        return jsonify({"success": False, "message": "Invalid data provided"}), 400
    except Exception:
        return jsonify({"success": False, "message": "An unexpected error occurred"}), 500
    return response


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/add_combiner", view_func=add_combiner, methods=["POST"])


def start_server_api():
    config = get_controller_config()
    port = config["port"]
    host = "0.0.0.0"
    debug = config["debug"]
    if debug:
        app.run(debug=debug, port=port, host=host)
    else:
        workers = os.cpu_count()
        gunicorn_app.run_gunicorn(app, host, port, workers)


if __name__ == "__main__":
    start_server_api()
