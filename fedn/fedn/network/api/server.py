import uuid

from flask import Flask, jsonify, request

from fedn.common.config import get_network_config, get_statestore_config
from fedn.network.api.interface import API
from fedn.network.controller.control import Control
from fedn.network.statestore.mongostatestore import MongoStateStore

statestore_config = get_statestore_config()
network_id = get_network_config()
statestore = MongoStateStore(
    network_id,
    statestore_config['mongo_config']
)
control = Control(statestore=statestore)
app = Flask(__name__)
api = API(statestore, control)


@app.route('/get_model_trail', methods=['GET'])
def get_model_trail():
    """ Get the model trail for a given session.
    param: session: The session id to get the model trail for.
    type: session: str
    return: The model trail for the given session as a json object.
    rtype: json
    """
    return api.get_model_trail()


@app.route('/delete_model_trail', methods=['GET', 'POST'])
def delete_model_trail():
    """ Delete the model trail for a given session.
    param: session: The session id to delete the model trail for.
    type: session: str
    return: The response from the statestore.
    rtype: json
    """
    return jsonify({"message": "Not implemented"}), 501


@app.route('/list_clients', methods=['GET'])
def list_clients():
    """ Get all clients from the statestore.
    return: All clients as a json object.
    rtype: json
    """
    return api.get_all_clients()


@app.route('/get_active_clients', methods=['GET'])
def get_active_clients():
    """ Get all active clients from the statestore.
    param: combiner_id: The combiner id to get active clients for.
    type: combiner_id: str
    return: All active clients as a json object.
    rtype: json
    """
    combiner_id = request.args.get('combiner', None)
    if combiner_id is None:
        return jsonify({"success": False, "message": "Missing combiner id."}), 400
    return api.get_active_clients(combiner_id)


@app.route('/list_combiners', methods=['GET'])
def list_combiners():
    """ Get all combiners in the network.
    return: All combiners as a json object.
    rtype: json
    """
    return api.get_all_combiners()


@app.route('/get_combiner', methods=['GET'])
def get_combiner():
    """ Get a combiner from the statestore.
    param: combiner_id: The combiner id to get.
    type: combiner_id: str
    return: The combiner as a json object.
    rtype: json
    """
    combiner_id = request.args.get('combiner', None)
    if combiner_id is None:
        return jsonify({"success": False, "message": "Missing combiner id."}), 400
    return api.get_combiner(combiner_id)


@app.route('/list_rounds', methods=['GET'])
def list_rounds():
    """ Get all rounds from the statestore.
    return: All rounds as a json object.
    rtype: json
    """
    return api.get_all_rounds()


@app.route('/get_round', methods=['GET'])
def get_round():
    """ Get a round from the statestore.
    param: round_id: The round id to get.
    type: round_id: str
    return: The round as a json object.
    rtype: json
    """
    round_id = request.args.get('round_id', None)
    if round_id is None:
        return jsonify({"success": False, "message": "Missing round id."}), 400
    return api.get_round(round_id)


@app.route('/start_session', methods=['GET', 'POST'])
def start_session():
    """ Start a new session.
    return: The response from control.
    rtype: json
    """
    # Get session id, if none is provided, generate a new one
    session_id = request.args.get('session_id', str(uuid.uuid4()))
    # Get round timeout, if none is provided, use default
    round_timeout = float(request.args.get('round_timeout', 180))
    # Get number of rounds, if none is provided, use default
    rounds = int(request.args.get('rounds', 5))
    # Get round buffer size, if none is provided, use default
    round_buffer_size = int(request.args.get('round_buffer_size', -1))
    # Get delete models, if none is provided, use default
    delete_models = request.args.get('delete_models', True)
    # Convert string to boolean, "True" -> True, "False" -> False
    if isinstance(delete_models, str):
        delete_models = delete_models == "True"
    # Get validation strategy, if none is provided, use default
    validation_strategy = request.args.get('validate', True)
    if isinstance(validation_strategy, str):
        validation_strategy = validation_strategy == "True"
    # Get helper type, if none is provided, use default
    helper_type = request.args.get('helper', 'kerashelper')
    # Get minimum number of clients, if none is provided, use default
    min_clients = int(request.args.get('min_clients', 1))
    # Get requested clients, if none is provided, use default
    requested_clients = int(request.args.get('requested_clients', 1))

    return api.start_session(session_id=session_id,
                             round_timeout=round_timeout,
                             rounds=rounds,
                             round_buffer_size=round_buffer_size,
                             delete_models=delete_models,
                             validate=validation_strategy,
                             helper=helper_type,
                             min_clients=min_clients,
                             requested_clients=requested_clients)


@app.route('/list_sessions', methods=['GET'])
def list_sessions():
    """ Get all sessions from the statestore.
    return: All sessions as a json object.
    rtype: json
    """
    return api.get_all_sessions()


@app.route('/get_session', methods=['GET'])
def get_session():
    """ Get a session from the statestore.
    param: session_id: The session id to get.
    type: session_id: str
    return: The session as a json object.
    rtype: json
    """
    session_id = request.args.get('session_id', None)
    if session_id is None:
        return jsonify({"success": False, "message": "Missing session id."}), 400
    return api.get_session(session_id)


@app.route('/set_package', methods=['POST'])
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
    helper_type = request.form.get('helper', None)
    if helper_type is None:
        return jsonify({"success": False, "message": "Missing helper type."}), 400
    try:
        file = request.files['file']
    except KeyError:
        return jsonify({"success": False, "message": "Missing file."}), 400
    return api.set_compute_package(file=file, helper_type=helper_type)


@app.route('/get_package', methods=['GET'])
def get_package():
    """ Get the compute package from the statestore.
    return: The compute package as a json object.
    rtype: json
    """
    return api.get_compute_package()


@app.route('/download_package', methods=['GET'])
def download_package():
    """ Download the compute package.
    return: The compute package as a json object.
    rtype: json
    """
    name = request.args.get('name', None)
    return api.download_compute_package(name)


@app.route('/get_package_checksum', methods=['GET'])
def get_package_checksum():
    name = request.args.get('name', None)
    return api.get_checksum(name)


@app.route('/get_latest_model', methods=['GET'])
def get_latest_model():
    """ Get the latest model from the statestore.
    return: The initial model as a json object.
    rtype: json
    """
    return api.get_latest_model()

# Get initial model endpoint


@app.route('/get_initial_model', methods=['GET'])
def get_initial_model():
    """ Get the initial model from the statestore.
    return: The initial model as a json object.
    rtype: json
    """
    return api.get_initial_model()


@app.route('/set_initial_model', methods=['POST'])
def set_initial_model():
    """ Set the initial model in the statestore and upload to model repository.
        Usage with curl:
        curl -k -X POST \
            -F file=@seed.npz \   
            http://localhost:8092/set_initial_model

    param: file: The initial model file to set.
    type: file: file
    return: The response from the statestore.
    rtype: json
    """
    try:
        file = request.files['file']
    except KeyError:
        return jsonify({"success": False, "message": "Missing file."}), 400
    return api.set_initial_model(file)


@app.route('/get_controller_status', methods=['GET'])
def get_controller_status():
    """ Get the status of the controller.
    return: The status as a json object.
    rtype: json
    """
    return api.get_controller_status()


@app.route('/get_events', methods=['GET'])
def get_events():
    """ Get the events from the statestore.
    return: The events as a json object.
    rtype: json
    """
    # TODO: except filter with request.get_json()
    kwargs = request.args.to_dict()
    return api.get_events(**kwargs)


@app.route('/list_validations', methods=['GET'])
def list_validations():
    """ Get all validations from the statestore.
    return: All validations as a json object.
    rtype: json
    """
    # TODO: except filter with request.get_json()
    kwargs = request.args.to_dict()
    return api.get_all_validations(**kwargs)


@app.route('/add_combiner', methods=['POST'])
def add_combiner():
    """ Add a combiner to the network.
    return: The response from the statestore.
    rtype: json
    """

    # name = request.args.get('name', None)
    # address = str(request.args.get('address', None))
    # remote_addr = request.remote_addr
    # fqdn = str(request.args.get('fqdn', None))
    # port = request.args.get('port', None)
    # secure_grpc = request.args.get('secure', None)

    json_data = request.get_json()
    remote_addr = request.remote_addr
    try:
        response = api.add_combiner(**json_data, remote_addr=remote_addr)
    except TypeError as e:
        return jsonify({'success': False, 'message': str(e)}), 400
    return response


@app.route('/add_client', methods=['POST'])
def add_client():
    """ Add a client to the network.
    return: The response from control.
    rtype: json
    """

    json_data = request.get_json()
    remote_addr = request.remote_addr
    try:
        response = api.add_client(**json_data, remote_addr=remote_addr)
    except TypeError as e:
        return jsonify({'success': False, 'message': str(e)}), 400
    return response


if __name__ == '__main__':
    app.run(debug=True, port=8092, host='0.0.0.0')
