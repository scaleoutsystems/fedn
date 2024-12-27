import os

from flask import Flask, jsonify, request

from fedn.common.config import get_controller_config
from fedn.network.api import gunicorn_app
from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.interface import API
from fedn.network.api.shared import control, statestore
from fedn.network.api.v1 import _routes
from fedn.network.api.v1.graphql.schema import schema

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


@app.route("/get_model", methods=["GET"])
@jwt_auth_required(role="admin")
def get_model():
    """Get a model from the statestore.
    param: model: The model id to get.
    type: model: str
    return: The model as a json object.
    rtype: json
    """
    model = request.args.get("model", None)
    if model is None:
        return jsonify({"success": False, "message": "Missing model id."}), 400

    return api.get_model(model)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_model", view_func=get_model, methods=["GET"])


@app.route("/delete_model_trail", methods=["GET", "POST"])
@jwt_auth_required(role="admin")
def delete_model_trail():
    """Delete the model trail for a given session.
    param: session: The session id to delete the model trail for.
    type: session: str
    return: The response from the statestore.
    rtype: json
    """
    return jsonify({"message": "Not implemented"}), 501


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/delete_model_trail", view_func=delete_model_trail, methods=["GET", "POST"])


@app.route("/list_clients", methods=["GET"])
@jwt_auth_required(role="admin")
def list_clients():
    """Get all clients from the statestore.
    return: All clients as a json object.
    rtype: json
    """
    limit = request.args.get("limit", None)
    skip = request.args.get("skip", None)
    status = request.args.get("status", None)

    return api.get_clients(limit, skip, status)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/list_clients", view_func=list_clients, methods=["GET"])


@app.route("/get_active_clients", methods=["GET"])
@jwt_auth_required(role="admin")
def get_active_clients():
    """Get all active clients from the statestore.
    param: combiner_id: The combiner id to get active clients for.
    type: combiner_id: str
    return: All active clients as a json object.
    rtype: json
    """
    combiner_id = request.args.get("combiner", None)
    if combiner_id is None:
        return (
            jsonify({"success": False, "message": "Missing combiner id."}),
            400,
        )
    return api.get_active_clients(combiner_id)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_active_clients", view_func=get_active_clients, methods=["GET"])


@app.route("/list_combiners", methods=["GET"])
@jwt_auth_required(role="admin")
def list_combiners():
    """Get all combiners in the network.
    return: All combiners as a json object.
    rtype: json
    """
    limit = request.args.get("limit", None)
    skip = request.args.get("skip", None)

    return api.get_all_combiners(limit, skip)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/list_combiners", view_func=list_combiners, methods=["GET"])


@app.route("/get_combiner", methods=["GET"])
@jwt_auth_required(role="admin")
def get_combiner():
    """Get a combiner from the statestore.
    param: combiner_id: The combiner id to get.
    type: combiner_id: str
    return: The combiner as a json object.
    rtype: json
    """
    combiner_id = request.args.get("combiner", None)
    if combiner_id is None:
        return (
            jsonify({"success": False, "message": "Missing combiner id."}),
            400,
        )
    return api.get_combiner(combiner_id)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_combiner", view_func=get_combiner, methods=["GET"])


@app.route("/get_package", methods=["GET"])
@jwt_auth_required(role="admin")
def get_package():
    """Get the compute package from the statestore.
    return: The compute package as a json object.
    rtype: json
    """
    return api.get_compute_package()


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_package", view_func=get_package, methods=["GET"])


@app.route("/download_package", methods=["GET"])
@jwt_auth_required(role="client")
def download_package():
    """Download the compute package.
    return: The compute package as a json object.
    rtype: json
    """
    name = request.args.get("name", None)
    return api.download_compute_package(name)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/download_package", view_func=download_package, methods=["GET"])


@app.route("/get_package_checksum", methods=["GET"])
@jwt_auth_required(role="client")
def get_package_checksum():
    name = request.args.get("name", None)
    return api.get_checksum(name)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_package_checksum", view_func=get_package_checksum, methods=["GET"])


@app.route("/set_initial_model", methods=["POST"])
@jwt_auth_required(role="admin")
def set_initial_model():
    """Set the initial model in the statestore and upload to model repository.
        Usage with curl:
        curl -k -X POST
            -F file=@seed.npz
            http://localhost:8092/set_initial_model

    param: file: The initial model file to set.
    type: file: file
    return: The response from the statestore.
    rtype: json
    """
    try:
        file = request.files["file"]
    except KeyError:
        return jsonify({"success": False, "message": "Missing file."}), 400
    return api.set_initial_model(file)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/set_initial_model", view_func=set_initial_model, methods=["POST"])


@app.route("/get_controller_status", methods=["GET"])
@jwt_auth_required(role="admin")
def get_controller_status():
    """Get the status of the controller.
    return: The status as a json object.
    rtype: json
    """
    return api.get_controller_status()


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


@app.route("/add_client", methods=["POST"])
@jwt_auth_required(role="client")
def add_client():
    """Add a client to the network.
    return: The response from control.
    rtype: json
    """
    json_data = request.get_json()
    remote_addr = request.remote_addr
    try:
        response = api.add_client(**json_data, remote_addr=remote_addr)
    except TypeError as e:
        print(e)
        return jsonify({"success": False, "message": "Invalid data provided"}), 400
    except Exception as e:
        print(e)
        return jsonify({"success": False, "message": "An unexpected error occurred"}), 500
    return response


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/add_client", view_func=add_client, methods=["POST"])


@app.route("/list_combiners_data", methods=["POST"])
@jwt_auth_required(role="admin")
def list_combiners_data():
    """List data from combiners.
    return: The response from control.
    rtype: json
    """
    json_data = request.get_json()

    # expects a list of combiner names (strings) in an array
    combiners = json_data.get("combiners", None)

    try:
        response = api.list_combiners_data(combiners)
    except TypeError:
        return jsonify({"success": False, "message": "Invalid data provided"}), 400
    except Exception:
        return jsonify({"success": False, "message": "An unexpected error occurred"}), 500
    return response


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/list_combiners_data", view_func=list_combiners_data, methods=["POST"])


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
