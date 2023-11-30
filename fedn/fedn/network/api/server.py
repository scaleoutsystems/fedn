from flask import Flask, jsonify, request

from fedn.common.config import (get_controller_config, get_modelstorage_config,
                                get_network_config, get_statestore_config)
from fedn.common.log_config import logger
from fedn.network.api.interface import API
from fedn.network.controller.control import Control
from fedn.network.statestore.mongostatestore import MongoStateStore


class Controller():
    def __init__(self):
        self.app = Flask(__name__)
        try:
            statestore_config = get_statestore_config()
        except FileNotFoundError as err:
            logger.debug("No statestore config, using default values.")
            statestore_config = {
                "type": "MongoDB",
                "mongo_config": {
                    "username": "admin",
                    "password": "admin",
                    "host": "localhost",
                    "port": 27017
                }
            }
        try:
            network_id = get_network_config()
        except FileNotFoundError as err:
            logger.debug("No network config found, using default values.")
            network_id = "fedn_network"
        try:
            modelstorage_config = get_modelstorage_config()
        except FileNotFoundError as err:
            logger.debug("No model storage config found, using default values.")
            modelstorage_config = {
                "storage_type": "filesystem",
                "storage_config": {
                    "storage_hostname": "localhost",
                    "storage_port": 9100,
                    "storage_access_key": "admin",
                    "storage_secret_key": "password",
                    "storage_bucket": "fedn-models",
                    "context_bucket": "fedn-context",
                    "storage_secure_mode": False 
                }
            }
        statestore = MongoStateStore(
            network_id, statestore_config["mongo_config"], modelstorage_config
        )
        control = Control(statestore=statestore)
        self.api = API(statestore, control)

        self.setup_routes()
        
    def setup_routes(self):
        @self.app.route("/get_model_trail", methods=["GET"])
        def get_model_trail():
            """Get the model trail for a given session.
            param: session: The session id to get the model trail for.
            type: session: str
            return: The model trail for the given session as a json object.
            rtype: json
            """
            return self.api.get_model_trail()


        @self.app.route("/list_models", methods=["GET"])
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

            return self.api.get_models(session_id, limit, skip)


        @self.app.route("/delete_model_trail", methods=["GET", "POST"])
        def delete_model_trail():
            """Delete the model trail for a given session.
            param: session: The session id to delete the model trail for.
            type: session: str
            return: The response from the statestore.
            rtype: json
            """
            return jsonify({"message": "Not implemented"}), 501


        @self.app.route("/list_clients", methods=["GET"])
        def list_clients():
            """Get all clients from the statestore.
            return: All clients as a json object.
            rtype: json
            """

            limit = request.args.get("limit", None)
            skip = request.args.get("skip", None)
            status = request.args.get("status", None)

            return self.api.get_clients(limit, skip, status)


        @self.app.route("/get_active_clients", methods=["GET"])
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
            return self.api.get_active_clients(combiner_id)


        @self.app.route("/list_combiners", methods=["GET"])
        def list_combiners():
            """Get all combiners in the network.
            return: All combiners as a json object.
            rtype: json
            """

            limit = request.args.get("limit", None)
            skip = request.args.get("skip", None)

            return self.api.get_all_combiners(limit, skip)


        @self.app.route("/get_combiner", methods=["GET"])
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
            return self.api.get_combiner(combiner_id)


        @self.app.route("/list_rounds", methods=["GET"])
        def list_rounds():
            """Get all rounds from the statestore.
            return: All rounds as a json object.
            rtype: json
            """
            return self.api.get_all_rounds()


        @self.app.route("/get_round", methods=["GET"])
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
            return self.api.get_round(round_id)


        @self.app.route("/start_session", methods=["GET", "POST"])
        def start_session():
            """Start a new session.
            return: The response from control.
            rtype: json
            """
            json_data = request.get_json()
            return self.api.start_session(**json_data)


        @self.app.route("/list_sessions", methods=["GET"])
        def list_sessions():
            """Get all sessions from the statestore.
            return: All sessions as a json object.
            rtype: json
            """
            limit = request.args.get("limit", None)
            skip = request.args.get("skip", None)

            return self.api.get_all_sessions(limit, skip)


        @self.app.route("/get_session", methods=["GET"])
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
            return self.api.get_session(session_id)


        @self.app.route("/set_package", methods=["POST"])
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
                print(file)
                print(file.content_length)
                file.seek(0, 2) # seeks the end of the file
                filesize = file.tell() # tell at which byte we are
                print(filesize)
                file.seek(0)
                file.save("testest.tgz")
            except KeyError:
                return jsonify({"success": False, "message": "Missing file."}), 400
            return self.api.set_compute_package(file=file, helper_type=helper_type)


        @self.app.route("/get_package", methods=["GET"])
        def get_package():
            """Get the compute package from the statestore.
            return: The compute package as a json object.
            rtype: json
            """
            return self.api.get_compute_package()


        @self.app.route("/download_package", methods=["GET"])
        def download_package():
            """Download the compute package.
            return: The compute package as a json object.
            rtype: json
            """
            print("HERE")
            name = request.args.get("name", None)
            print(name)
            return self.api.download_compute_package(name)


        @self.app.route("/get_package_checksum", methods=["GET"])
        def get_package_checksum():
            name = request.args.get("name", None)
            return self.api.get_checksum(name)


        @self.app.route("/get_latest_model", methods=["GET"])
        def get_latest_model():
            """Get the latest model from the statestore.
            return: The initial model as a json object.
            rtype: json
            """
            return self.api.get_latest_model()


        # Get initial model endpoint


        @self.app.route("/get_initial_model", methods=["GET"])
        def get_initial_model():
            """Get the initial model from the statestore.
            return: The initial model as a json object.
            rtype: json
            """
            return self.api.get_initial_model()


        @self.app.route("/set_initial_model", methods=["POST"])
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
            return self.api.set_initial_model(file)


        @self.app.route("/get_controller_status", methods=["GET"])
        def get_controller_status():
            """Get the status of the controller.
            return: The status as a json object.
            rtype: json
            """
            return self.api.get_controller_status()


        @self.app.route("/get_client_config", methods=["GET"])
        def get_client_config():
            """Get the client configuration.
            return: The client configuration as a json object.
            rtype: json
            """
            checksum = request.args.get("checksum", True)
            return self.api.get_client_config(checksum)


        @self.app.route("/get_events", methods=["GET"])
        def get_events():
            """Get the events from the statestore.
            return: The events as a json object.
            rtype: json
            """
            # TODO: except filter with request.get_json()
            kwargs = request.args.to_dict()

            return self.api.get_events(**kwargs)


        @self.app.route("/list_validations", methods=["GET"])
        def list_validations():
            """Get all validations from the statestore.
            return: All validations as a json object.
            rtype: json
            """
            # TODO: except filter with request.get_json()
            kwargs = request.args.to_dict()
            return self.api.get_all_validations(**kwargs)


        @self.app.route("/add_combiner", methods=["POST"])
        def add_combiner():
            """Add a combiner to the network.
            return: The response from the statestore.
            rtype: json
            """
            json_data = request.get_json()
            remote_addr = request.remote_addr
            try:
                response = self.api.add_combiner(**json_data, remote_addr=remote_addr)
            except TypeError as e:
                return jsonify({"success": False, "message": str(e)}), 400
            return response


        @self.app.route("/add_client", methods=["POST"])
        def add_client():
            """Add a client to the network.
            return: The response from control.
            rtype: json
            """

            json_data = request.get_json()
            remote_addr = request.remote_addr
            try:
                response = self.api.add_client(**json_data, remote_addr=remote_addr)
            except TypeError as e:
                return jsonify({"success": False, "message": str(e)}), 400
            return response


        @self.app.route("/list_combiners_data", methods=["POST"])
        def list_combiners_data():
            """List data from combiners.
            return: The response from control.
            rtype: json
            """

            json_data = request.get_json()

            # expects a list of combiner names (strings) in an array
            combiners = json_data.get("combiners", None)

            try:
                response = self.api.list_combiners_data(combiners)
            except TypeError as e:
                return jsonify({"success": False, "message": str(e)}), 400
            return response


        @self.app.route("/get_plot_data", methods=["GET"])
        def get_plot_data():
            """Get plot data from the statestore.
            rtype: json
            """

            try:
                feature = request.args.get("feature", None)
                response = self.api.get_plot_data(feature=feature)
            except TypeError as e:
                return jsonify({"success": False, "message": str(e)}), 400
            return response

    def run(self):
        try:
            config = get_controller_config()
        except FileNotFoundError as err:
            logger.debug("Found no controller config, using default values.")
            config = {
                "host": "localhost",
                "port": 8092,
                "debug": True
            }
        port = config["port"]
        debug = config["debug"]
        self.app.run(debug=debug, port=port, host="0.0.0.0")

if __name__ == "__main__":
    controller = Controller()
    controller.run()
