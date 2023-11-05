from flask import Flask, jsonify, request

from fedn.common.config import (get_controller_config, get_modelstorage_config,
                                get_network_config, get_statestore_config)
from fedn.network.api.interface import API
from fedn.network.controller.control import Control
from fedn.network.statestore.mongostatestore import MongoStateStore

statestore_config = get_statestore_config()
network_id = get_network_config()
modelstorage_config = get_modelstorage_config()
statestore = MongoStateStore(
    network_id, statestore_config["mongo_config"], modelstorage_config
)
control = Control(statestore=statestore)
api = API(statestore, control)
app = Flask(__name__)


@app.route("/get_model_trail", methods=["GET"])
def get_model_trail():
    """Get the model trail for a given session.
    param: session: The session id to get the model trail for.
    type: session: str
    return: The model trail for the given session as a json object.
    rtype: json
    """
    return api.get_model_trail()


@app.route("/list_models", methods=["GET"])
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

    return api.get_models(session_id, limit, skip)


@app.route("/delete_model_trail", methods=["GET", "POST"])
def delete_model_trail():
    """Delete the model trail for a given session.
    param: session: The session id to delete the model trail for.
    type: session: str
    return: The response from the statestore.
    rtype: json
    """
    return jsonify({"message": "Not implemented"}), 501


@app.route("/list_clients", methods=["GET"])
def list_clients():
    """Get all clients from the statestore.
    return: All clients as a json object.
    rtype: json
    """

    limit = request.args.get("limit", None)
    skip = request.args.get("skip", None)
    status = request.args.get("status", None)

    return api.get_clients(limit, skip, status)


@app.route("/get_active_clients", methods=["GET"])
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


@app.route("/list_combiners", methods=["GET"])
def list_combiners():
    """Get all combiners in the network.
    return: All combiners as a json object.
    rtype: json
    """

    limit = request.args.get("limit", None)
    skip = request.args.get("skip", None)

    return api.get_all_combiners(limit, skip)


@app.route("/get_combiner", methods=["GET"])
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


@app.route("/list_rounds", methods=["GET"])
def list_rounds():
    """Get all rounds from the statestore.
    return: All rounds as a json object.
    rtype: json
    """
    return api.get_all_rounds()


@app.route("/get_round", methods=["GET"])
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


@app.route("/start_session", methods=["GET", "POST"])
def start_session():
    """Start a new session.
    return: The response from control.
    rtype: json
    """
    json_data = request.get_json()
    return api.start_session(**json_data)


@app.route("/list_sessions", methods=["GET"])
def list_sessions():
    """Get all sessions from the statestore.
    return: All sessions as a json object.
    rtype: json
    """
    limit = request.args.get("limit", None)
    skip = request.args.get("skip", None)

    return api.get_all_sessions(limit, skip)


@app.route("/get_session", methods=["GET"])
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


@app.route("/set_package", methods=["POST"])
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
    if helper_type is None:
        return (
            jsonify({"success": False, "message": "Missing helper type."}),
            400,
        )
    try:
        file = request.files["file"]
    except KeyError:
        return jsonify({"success": False, "message": "Missing file."}), 400
    return api.set_compute_package(file=file, helper_type=helper_type)


@app.route("/get_package", methods=["GET"])
def get_package():
    """Get the compute package from the statestore.
    return: The compute package as a json object.
    rtype: json
    """
    return api.get_compute_package()


@app.route("/download_package", methods=["GET"])
def download_package():
    """Download the compute package.
    return: The compute package as a json object.
    rtype: json
    """
    name = request.args.get("name", None)
    return api.download_compute_package(name)


@app.route("/get_package_checksum", methods=["GET"])
def get_package_checksum():
    name = request.args.get("name", None)
    return api.get_checksum(name)


@app.route("/get_latest_model", methods=["GET"])
def get_latest_model():
    """Get the latest model from the statestore.
    return: The initial model as a json object.
    rtype: json
    """
    return api.get_latest_model()


# Get initial model endpoint


@app.route("/get_initial_model", methods=["GET"])
def get_initial_model():
    """Get the initial model from the statestore.
    return: The initial model as a json object.
    rtype: json
    """
    return api.get_initial_model()


@app.route("/set_initial_model", methods=["POST"])
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


@app.route("/get_controller_status", methods=["GET"])
def get_controller_status():
    """Get the status of the controller.
    return: The status as a json object.
    rtype: json
    """
    return api.get_controller_status()


@app.route("/get_client_config", methods=["GET"])
def get_client_config():
    """Get the client configuration.
    return: The client configuration as a json object.
    rtype: json
    """
    checksum = request.args.get("checksum", True)
    return api.get_client_config(checksum)


@app.route("/get_events", methods=["GET"])
def get_events():
    """Get the events from the statestore.
    return: The events as a json object.
    rtype: json
    """
    # TODO: except filter with request.get_json()
    kwargs = request.args.to_dict()

    return api.get_events(**kwargs)


@app.route("/list_validations", methods=["GET"])
def list_validations():
    """Get all validations from the statestore.
    return: All validations as a json object.
    rtype: json
    """
    # TODO: except filter with request.get_json()
    kwargs = request.args.to_dict()
    return api.get_all_validations(**kwargs)


@app.route("/add_combiner", methods=["POST"])
def add_combiner():
    """Add a combiner to the network.
    return: The response from the statestore.
    rtype: json
    """
    json_data = request.get_json()
    remote_addr = request.remote_addr
    try:
        response = api.add_combiner(**json_data, remote_addr=remote_addr)
    except TypeError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    return response


@app.route("/add_client", methods=["POST"])
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
        return jsonify({"success": False, "message": str(e)}), 400
    return response


@app.route("/list_combiners_data", methods=["POST"])
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
    except TypeError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    return response


@app.route("/get_plot_data", methods=["GET"])
def get_plot_data():
    """Get plot data from the statestore.
    rtype: json
    """

    try:
        feature = request.args.get("feature", None)
        response = api.get_plot_data(feature=feature)
    except TypeError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    return response


if __name__ == "__main__":
    config = get_controller_config()
    port = config["port"]
    debug = config["debug"]
    app.run(debug=debug, port=port, host="0.0.0.0")
