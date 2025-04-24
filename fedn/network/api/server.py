import os

from flask import Flask, jsonify, request

from fedn.common.config import get_controller_config, get_modelstorage_config, get_network_config, get_statestore_config
from fedn.network.api import gunicorn_app
from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.v1 import _routes
from fedn.network.api.v1.graphql.schema import schema
from fedn.network.controller.control import Control
from fedn.network.state import ReducerStateToString
from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.s3.repository import Repository

custom_url_prefix = os.environ.get("FEDN_CUSTOM_URL_PREFIX", False)
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
    return jsonify({"state": ReducerStateToString(Control.instance().state())}), 200


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_controller_status", view_func=get_controller_status, methods=["GET"])


@app.route("/add_combiner", methods=["POST"])
@jwt_auth_required(role="combiner")
def add_combiner():
    """Add a combiner to the network.
    return: The response from the statestore.
    rtype: json
    """
    payload = {
        "success": False,
        "message": "Adding combiner via REST API is obsolete. Include statestore and object store config in combiner config.",
        "status": "abort",
    }

    return jsonify(payload), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/add_combiner", view_func=add_combiner, methods=["POST"])


# deprecated endpoints


@app.route("/get_model_trail", methods=["GET"])
@jwt_auth_required(role="admin")
def get_model_trail():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/models or the GraphQL API instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_model_trail", view_func=get_model_trail, methods=["GET"])


@app.route("/get_model_ancestors", methods=["GET"])
@jwt_auth_required(role="admin")
def get_model_ancestors():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/models/<id>/ancestors or the GraphQL API instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_model_ancestors", view_func=get_model_ancestors, methods=["GET"])


@app.route("/get_model_descendants", methods=["GET"])
@jwt_auth_required(role="admin")
def get_model_descendants():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/models/<id>/descendants or the GraphQL API instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_model_descendants", view_func=get_model_descendants, methods=["GET"])


@app.route("/list_models", methods=["GET"])
@jwt_auth_required(role="admin")
def list_models():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/models or the GraphQL API instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/list_models", view_func=list_models, methods=["GET"])


@app.route("/get_model", methods=["GET"])
@jwt_auth_required(role="admin")
def get_model():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/models/<id> or the GraphQL API instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_model", view_func=get_model, methods=["GET"])


@app.route("/list_clients", methods=["GET"])
@jwt_auth_required(role="admin")
def list_clients():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/clients instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/list_clients", view_func=list_clients, methods=["GET"])


@app.route("/get_active_clients", methods=["GET"])
@jwt_auth_required(role="admin")
def get_active_clients():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/clients instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_active_clients", view_func=get_active_clients, methods=["GET"])


@app.route("/list_combiners", methods=["GET"])
@jwt_auth_required(role="admin")
def list_combiners():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/combiners instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/list_combiners", view_func=list_combiners, methods=["GET"])


@app.route("/get_combiner", methods=["GET"])
@jwt_auth_required(role="admin")
def get_combiner():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/combiners/<id> instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_combiner", view_func=get_combiner, methods=["GET"])


@app.route("/list_rounds", methods=["GET"])
@jwt_auth_required(role="admin")
def list_rounds():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/rounds instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/list_rounds", view_func=list_rounds, methods=["GET"])


@app.route("/get_round", methods=["GET"])
@jwt_auth_required(role="admin")
def get_round():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/rounds/<id> instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_round", view_func=get_round, methods=["GET"])


@app.route("/start_session", methods=["GET", "POST"])
@jwt_auth_required(role="admin")
def start_session():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/sessions and /api/v1/sessions/start instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/start_session", view_func=start_session, methods=["GET", "POST"])


@app.route("/list_sessions", methods=["GET"])
@jwt_auth_required(role="admin")
def list_sessions():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/sessions or the GraphQL API instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/list_sessions", view_func=list_sessions, methods=["GET"])


@app.route("/get_session", methods=["GET"])
@jwt_auth_required(role="admin")
def get_session():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/sessions<id> or the GraphQL API instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_session", view_func=get_session, methods=["GET"])


@app.route("/set_active_package", methods=["PUT"])
@jwt_auth_required(role="admin")
def set_active_package():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/packages/active instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/set_active_package", view_func=set_active_package, methods=["PUT"])


@app.route("/set_package", methods=["POST"])
@jwt_auth_required(role="admin")
def set_package():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/packages instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/set_package", view_func=set_package, methods=["POST"])


@app.route("/get_package", methods=["GET"])
@jwt_auth_required(role="admin")
def get_package():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/packages/active or /api/v1/packages/<id> instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_package", view_func=get_package, methods=["GET"])


@app.route("/list_compute_packages", methods=["GET"])
@jwt_auth_required(role="admin")
def list_compute_packages():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/packages instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/list_compute_packages", view_func=list_compute_packages, methods=["GET"])


@app.route("/download_package", methods=["GET"])
@jwt_auth_required(role="client")
def download_package():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/packages/download instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/download_package", view_func=download_package, methods=["GET"])


@app.route("/get_package_checksum", methods=["GET"])
@jwt_auth_required(role="client")
def get_package_checksum():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/packages/checksum instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_package_checksum", view_func=get_package_checksum, methods=["GET"])


@app.route("/get_latest_model", methods=["GET"])
@jwt_auth_required(role="admin")
def get_latest_model():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/models instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_latest_model", view_func=get_latest_model, methods=["GET"])


@app.route("/set_current_model", methods=["PUT"])
@jwt_auth_required(role="admin")
def set_current_model():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/models instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/set_current_model", view_func=set_current_model, methods=["PUT"])


@app.route("/get_initial_model", methods=["GET"])
@jwt_auth_required(role="admin")
def get_initial_model():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/models instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_initial_model", view_func=get_initial_model, methods=["GET"])


@app.route("/set_initial_model", methods=["POST"])
@jwt_auth_required(role="admin")
def set_initial_model():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/models instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/set_initial_model", view_func=set_initial_model, methods=["POST"])


@app.route("/get_client_config", methods=["GET"])
@jwt_auth_required(role="admin")
def get_client_config():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/clients/config instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_client_config", view_func=get_client_config, methods=["GET"])


@app.route("/get_events", methods=["GET"])
@jwt_auth_required(role="admin")
def get_events():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/statuses instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_events", view_func=get_events, methods=["GET"])


@app.route("/list_validations", methods=["GET"])
@jwt_auth_required(role="admin")
def list_validations():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/validations or the GraphQL API instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/list_validations", view_func=list_validations, methods=["GET"])


@app.route("/add_client", methods=["POST"])
@jwt_auth_required(role="client")
def add_client():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/clients/add instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/add_client", view_func=add_client, methods=["POST"])


@app.route("/list_combiners_data", methods=["POST"])
@jwt_auth_required(role="admin")
def list_combiners_data():
    response = {
        "message": "This endpoint is deprecated. Use /api/v1/combiners/clients/count instead.",
    }
    return jsonify(response), 410


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/list_combiners_data", view_func=list_combiners_data, methods=["POST"])

# not implemented


@app.route("/delete_model_trail", methods=["GET", "POST"])
@jwt_auth_required(role="admin")
def delete_model_trail():
    return jsonify({"message": "Not implemented"}), 501


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/delete_model_trail", view_func=delete_model_trail, methods=["GET", "POST"])


def start_server_api():
    config = get_controller_config()
    port = config["port"]
    host = "0.0.0.0"
    debug = config["debug"]

    network_id = get_network_config()
    modelstorage_config = get_modelstorage_config()
    statestore_config = get_statestore_config()

    # TODO: Initialize database with config instead of reading it under the hood
    db = DatabaseConnection(statestore_config, network_id)
    repository = Repository(modelstorage_config["storage_config"], storage_type=modelstorage_config["storage_type"])
    Control.create_instance(network_id, repository, db)

    if debug:
        app.run(debug=debug, port=port, host=host)
    else:
        workers = os.cpu_count()
        gunicorn_app.run_gunicorn(app, host, port, workers)


if __name__ == "__main__":
    start_server_api()
