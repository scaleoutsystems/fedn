import os

from flask import Flask, jsonify, request

from fedn.common.config import get_controller_config
from fedn.network.api.auth import jwt_auth_required
from fedn.network.api.interface import API
from fedn.network.api.shared import control, statestore
from fedn.network.api.v1 import _routes
from fedn.network.api import gunicorn_app

custom_url_prefix = os.environ.get("FEDN_CUSTOM_URL_PREFIX", False)
# statestore_config,modelstorage_config,network_id,control=set_statestore_config()
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


@app.route("/get_model_trail", methods=["GET"])
@jwt_auth_required(role="admin")
def get_model_trail():
    """Get the model trail for a given session.
    param: session: The session id to get the model trail for.
    type: session: str
    return: The model trail for the given session as a json object.
    rtype: json
    """
    return api.get_model_trail()


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_model_trail", view_func=get_model_trail, methods=["GET"])


@app.route("/get_model_ancestors", methods=["GET"])
@jwt_auth_required(role="admin")
def get_model_ancestors():
    """Get the ancestors of a model.
    param: model: The model id to get the ancestors for.
    type: model: str
    param: limit: The maximum number of ancestors to return.
    type: limit: int
    return: A list of model objects that the model derives from.
    rtype: json
    """
    model = request.args.get("model", None)
    limit = request.args.get("limit", None)

    return api.get_model_ancestors(model, limit)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_model_ancestors", view_func=get_model_ancestors, methods=["GET"])


@app.route("/get_model_descendants", methods=["GET"])
@jwt_auth_required(role="admin")
def get_model_descendants():
    """Get the ancestors of a model.
    param: model: The model id to get the child for.
    type: model: str
    param: limit: The maximum number of descendants to return.
    type: limit: int
    return: A list of model objects that are descendents of the provided model id.
    rtype: json
    """
    model = request.args.get("model", None)
    limit = request.args.get("limit", None)

    return api.get_model_descendants(model, limit)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_model_descendants", view_func=get_model_descendants, methods=["GET"])


@app.route("/list_models", methods=["GET"])
@jwt_auth_required(role="admin")
def list_models():
    """Get models from the statestore.
    param:
    session_id: The session id to get the model trail for.
    limit: The maximum number of models to return.
    type: limit: int
    param: skip: The number of models to skip.
    type: skip: int
    Returns:
        _type_: json
    """
    session_id = request.args.get("session_id", None)
    limit = request.args.get("limit", None)
    skip = request.args.get("skip", None)
    include_active = request.args.get("include_active", None)

    return api.get_models(session_id, limit, skip, include_active)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/list_models", view_func=list_models, methods=["GET"])


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


@app.route("/list_rounds", methods=["GET"])
@jwt_auth_required(role="admin")
def list_rounds():
    """Get all rounds from the statestore.
    return: All rounds as a json object.
    rtype: json
    """
    return api.get_all_rounds()


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/list_rounds", view_func=list_rounds, methods=["GET"])


@app.route("/get_round", methods=["GET"])
@jwt_auth_required(role="admin")
def get_round():
    """Get a round from the statestore.
    param: round_id: The round id to get.
    type: round_id: str
    return: The round as a json object.
    rtype: json
    """
    round_id = request.args.get("round_id", None)
    if round_id is None:
        return jsonify({"success": False, "message": "Missing round id."}), 400
    return api.get_round(round_id)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_round", view_func=get_round, methods=["GET"])


@app.route("/start_session", methods=["GET", "POST"])
@jwt_auth_required(role="admin")
def start_session():
    """Start a new session.
    return: The response from control.
    rtype: json
    """
    json_data = request.get_json()
    return api.start_session(**json_data)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/start_session", view_func=start_session, methods=["GET", "POST"])


@app.route("/list_sessions", methods=["GET"])
@jwt_auth_required(role="admin")
def list_sessions():
    """Get all sessions from the statestore.
    return: All sessions as a json object.
    rtype: json
    """
    limit = request.args.get("limit", None)
    skip = request.args.get("skip", None)

    return api.get_all_sessions(limit, skip)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/list_sessions", view_func=list_sessions, methods=["GET"])


@app.route("/get_session", methods=["GET"])
@jwt_auth_required(role="admin")
def get_session():
    """Get a session from the statestore.
    param: session_id: The session id to get.
    type: session_id: str
    return: The session as a json object.
    rtype: json
    """
    session_id = request.args.get("session_id", None)
    if session_id is None:
        return (
            jsonify({"success": False, "message": "Missing session id."}),
            400,
        )
    return api.get_session(session_id)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_session", view_func=get_session, methods=["GET"])


@app.route("/set_active_package", methods=["PUT"])
@jwt_auth_required(role="admin")
def set_active_package():
    id = request.args.get("id", None)
    return api.set_active_compute_package(id)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/set_active_package", view_func=set_active_package, methods=["PUT"])


@app.route("/set_package", methods=["POST"])
@jwt_auth_required(role="admin")
def set_package():
    """ Set the compute package in the statestore.
        Usage with curl:
        curl -k -X POST \
            -F file=@package.tgz \
            -F helper="kerashelper" \
            http://localhost:8092/set_package

    param: file: The compute package file to set.
    type: file: file
    return: The response from the statestore.
    rtype: json
    """
    helper_type = request.form.get("helper", None)
    name = request.form.get("name", None)
    description = request.form.get("description", None)

    if helper_type is None:
        return (
            jsonify({"success": False, "message": "Missing helper type."}),
            400,
        )
    try:
        file = request.files["file"]
    except KeyError:
        return jsonify({"success": False, "message": "Missing file."}), 400
    return api.set_compute_package(file=file, helper_type=helper_type, name=name, description=description)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/set_package", view_func=set_package, methods=["POST"])


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


@app.route("/list_compute_packages", methods=["GET"])
@jwt_auth_required(role="admin")
def list_compute_packages():
    """Get the compute package from the statestore.
    return: The compute package as a json object.
    rtype: json
    """
    limit = request.args.get("limit", None)
    skip = request.args.get("skip", None)
    include_active = request.args.get("include_active", None)

    return api.list_compute_packages(limit=limit, skip=skip, include_active=include_active)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/list_compute_packages", view_func=list_compute_packages, methods=["GET"])


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


@app.route("/get_latest_model", methods=["GET"])
@jwt_auth_required(role="admin")
def get_latest_model():
    """Get the latest model from the statestore.
    return: The initial model as a json object.
    rtype: json
    """
    return api.get_latest_model()


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_latest_model", view_func=get_latest_model, methods=["GET"])


@app.route("/set_current_model", methods=["PUT"])
@jwt_auth_required(role="admin")
def set_current_model():
    """Set the initial model in the statestore and upload to model repository.
        Usage with curl:
        curl -k -X PUT
            -F id=<model-id>
            http://localhost:8092/set_current_model

    param: id: The model id to set.
    type: id: str
    return: boolean.
    rtype: json
    """
    id = request.args.get("id", None)
    if id is None:
        return jsonify({"success": False, "message": "Missing model id."}), 400
    return api.set_current_model(id)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/set_current_model", view_func=set_current_model, methods=["PUT"])

# Get initial model endpoint


@app.route("/get_initial_model", methods=["GET"])
@jwt_auth_required(role="admin")
def get_initial_model():
    """Get the initial model from the statestore.
    return: The initial model as a json object.
    rtype: json
    """
    return api.get_initial_model()


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_initial_model", view_func=get_initial_model, methods=["GET"])


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


@app.route("/get_events", methods=["GET"])
@jwt_auth_required(role="admin")
def get_events():
    """Get the events from the statestore.
    return: The events as a json object.
    rtype: json
    """
    # TODO: except filter with request.get_json()
    kwargs = request.args.to_dict()

    return api.get_events(**kwargs)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/get_events", view_func=get_events, methods=["GET"])


@app.route("/list_validations", methods=["GET"])
@jwt_auth_required(role="admin")
def list_validations():
    """Get all validations from the statestore.
    return: All validations as a json object.
    rtype: json
    """
    # TODO: except filter with request.get_json()
    kwargs = request.args.to_dict()
    return api.get_all_validations(**kwargs)


if custom_url_prefix:
    app.add_url_rule(f"{custom_url_prefix}/list_validations", view_func=list_validations, methods=["GET"])


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
        workers=os.cpu_count()
        gunicorn_app.run_gunicorn(app, host, port, workers)
if __name__ == "__main__":
    start_server_api()
