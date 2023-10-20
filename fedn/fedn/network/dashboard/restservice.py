import base64
import copy
import datetime
import json
import os
import re
import threading
from io import BytesIO
from threading import Lock

import jwt
import pandas as pd
from bokeh.embed import json_item
from bson import json_util
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from fedn.common.tracer.mongotracer import MongoTracer
from fedn.network.combiner.interfaces import CombinerInterface
from fedn.network.dashboard.plots import Plot
from fedn.network.state import ReducerState, ReducerStateToString
from fedn.utils.checksum import sha

UPLOAD_FOLDER = "/app/client/package/"
ALLOWED_EXTENSIONS = {"gz", "bz2", "tar", "zip", "tgz"}


def allowed_file(filename):
    """

    :param filename:
    :return:
    """
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def encode_auth_token(secret_key):
    """Generates the Auth Token
    :return: string
    """
    try:
        payload = {
            "exp": datetime.datetime.utcnow()
            + datetime.timedelta(days=90, seconds=0),
            "iat": datetime.datetime.utcnow(),
            "status": "Success",
        }
        token = jwt.encode(payload, secret_key, algorithm="HS256")
        print(
            "\n\n\nSECURE MODE ENABLED, USE TOKEN TO ACCESS REDUCER: **** {} ****\n\n\n".format(
                token
            )
        )
        return token
    except Exception as e:
        return e


def decode_auth_token(auth_token, secret):
    """Decodes the auth token
    :param auth_token:
    :return: string
    """
    try:
        payload = jwt.decode(auth_token, secret, algorithms=["HS256"])
        return payload["status"]
    except jwt.ExpiredSignatureError as e:
        print(e)
        return "Token has expired."
    except jwt.InvalidTokenError as e:
        print(e)
        return "Invalid token."


class ReducerRestService:
    """ """

    def __init__(self, config, control, statestore, certificate_manager):
        print("config object!: \n\n\n\n{}".format(config))
        if config["host"]:
            self.host = config["host"]
        else:
            self.host = None

        self.name = config["name"]

        self.port = config["port"]
        self.network_id = config["name"] + "-network"

        if "token" in config.keys():
            self.token_auth_enabled = True
        else:
            self.token_auth_enabled = False

        if "secret_key" in config.keys():
            self.SECRET_KEY = config["secret_key"]
        else:
            self.SECRET_KEY = None

        if "use_ssl" in config.keys():
            self.use_ssl = config["use_ssl"]

        self.remote_compute_package = config["remote_compute_package"]
        if self.remote_compute_package:
            self.package = "remote"
        else:
            self.package = "local"

        self.control = control
        self.statestore = statestore
        self.certificate_manager = certificate_manager
        self.current_compute_context = None

    def to_dict(self):
        """

        :return:
        """
        data = {"name": self.name}
        return data

    def check_compute_package(self):
        """Check if the compute package has been configured,
        if remote compute context is set to False, True will be returned

        :return: True if configured
        :rtype: bool
        """
        if not self.remote_compute_package:
            return True

        if not self.control.get_compute_package():
            return False
        else:
            return True

    def check_initial_model(self):
        """Check if initial model (seed model) has been configured

        :return: True if configured, else False
        :rtype: bool
        """

        if self.statestore.get_latest_model():
            return True
        else:
            return False

    def check_configured_response(self):
        """Check if everything has been configured for client to connect,
        return response if not.

        :return: Reponse with message if not configured, else None
        :rtype: json
        """
        if self.control.state() == ReducerState.setup:
            return jsonify(
                {
                    "status": "retry",
                    "package": self.package,
                    "msg": "Controller is not configured.",
                }
            )

        if not self.check_compute_package():
            return jsonify(
                {
                    "status": "retry",
                    "package": self.package,
                    "msg": "Compute package is not configured. Please upload the compute package.",
                }
            )

        if not self.check_initial_model():
            return jsonify(
                {
                    "status": "retry",
                    "package": self.package,
                    "msg": "Initial model is not configured. Please upload the model.",
                }
            )

        if not self.control.idle():
            return jsonify(
                {
                    "status": "retry",
                    "package": self.package,
                    "msg": "Conroller is not in idle state, try again later. ",
                }
            )
        return None

    def check_configured(self):
        """Check if compute package has been configured and that and that the
        state of the ReducerControl is not in setup otherwise render setup template.
        Check if initial model has been configured, otherwise render setup_model template.
        :return: Rendered html template or None
        """
        if not self.check_compute_package():
            return render_template(
                "setup.html",
                client=self.name,
                state=ReducerStateToString(self.control.state()),
                logs=None,
                refresh=False,
                message="Please set the compute package",
            )

        if self.control.state() == ReducerState.setup:
            return render_template(
                "setup.html",
                client=self.name,
                state=ReducerStateToString(self.control.state()),
                logs=None,
                refresh=True,
                message="Warning. Reducer is not base-configured. please do so with config file.",
            )

        if not self.check_initial_model():
            return render_template(
                "setup_model.html", message="Please set the initial model."
            )

        return None

    def authorize(self, r, secret):
        """Authorize client token

        :param r: Request
        :type r: [type]
        :param token: Token to verify against
        :type token: string
        """
        try:
            # Get token
            if "Authorization" in r.headers:  # header auth
                request_token = r.headers.get("Authorization").split()[1]
            elif "token" in r.args:  # args auth
                request_token = str(r.args.get("token"))
            elif "fedn_token" in r.cookies:
                request_token = r.cookies.get("fedn_token")
            else:  # no token provided
                print("Authorization failed. No token provided.", flush=True)
                abort(401)

            # Log token and secret
            print(
                f"Secret: {secret}. Request token: {request_token}.",
                flush=True,
            )

            # Authenticate
            status = decode_auth_token(request_token, secret)
            if status == "Success":
                return True
            else:
                print(
                    'Authorization failed. Status: "{}"'.format(status),
                    flush=True,
                )
                abort(401)
        except Exception as e:
            print(
                'Authorization failed. Expection encountered: "{}".'.format(e),
                flush=True,
            )
            abort(401)

    def run(self):
        """

        :return:
        """
        app = Flask(__name__)

        app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
        app.config["SECRET_KEY"] = self.SECRET_KEY

        @app.route("/")
        def index():
            """

            :return:
            """
            # Token auth
            if self.token_auth_enabled:
                self.authorize(request, app.config.get("SECRET_KEY"))

            # Render template
            not_configured_template = self.check_configured()
            if not_configured_template:
                template = not_configured_template
            else:
                events = self.control.get_events()
                message = request.args.get("message", None)
                message_type = request.args.get("message_type", None)
                template = render_template(
                    "events.html",
                    client=self.name,
                    state=ReducerStateToString(self.control.state()),
                    events=events,
                    logs=None,
                    refresh=True,
                    configured=True,
                    message=message,
                    message_type=message_type,
                )

            # Set token cookie in response if needed
            response = make_response(template)
            if "token" in request.args:  # args auth
                response.set_cookie("fedn_token", str(request.args["token"]))

            # Return response
            return response

        @app.route("/status")
        def status():
            """

            :return:
            """
            return {"state": ReducerStateToString(self.control.state())}

        @app.route("/netgraph")
        def netgraph():
            """
            Creates nodes and edges for network graph

            :return: nodes and edges as keys
            :rtype: dict
            """
            result = {"nodes": [], "edges": []}

            result["nodes"].append(
                {
                    "id": "reducer",
                    "label": "Reducer",
                    "role": "reducer",
                    "status": "active",
                    "name": "reducer",  # TODO: get real host name
                    "type": "reducer",
                }
            )

            combiner_info = combiner_status()
            client_info = client_status()

            if len(combiner_info) < 1:
                return result

            for combiner in combiner_info:
                print("combiner info {}".format(combiner_info), flush=True)
                try:
                    result["nodes"].append(
                        {
                            "id": combiner["name"],  # "n{}".format(count),
                            "label": "Combiner ({} clients)".format(
                                combiner["nr_active_clients"]
                            ),
                            "role": "combiner",
                            "status": "active",  # TODO: Hard-coded, combiner_info does not contain status
                            "name": combiner["name"],
                            "type": "combiner",
                        }
                    )
                except Exception as err:
                    print(err)

            for client in client_info["active_clients"]:
                try:
                    if client["status"] != "offline":
                        result["nodes"].append(
                            {
                                "id": str(client["_id"]),
                                "label": "Client",
                                "role": client["role"],
                                "status": client["status"],
                                "name": client["name"],
                                "combiner": client["combiner"],
                                "type": "client",
                            }
                        )
                except Exception as err:
                    print(err)

            count = 0
            for node in result["nodes"]:
                try:
                    if node["type"] == "combiner":
                        result["edges"].append(
                            {
                                "id": "e{}".format(count),
                                "source": node["id"],
                                "target": "reducer",
                            }
                        )
                    elif node["type"] == "client":
                        result["edges"].append(
                            {
                                "id": "e{}".format(count),
                                "source": node["combiner"],
                                "target": node["id"],
                            }
                        )
                except Exception:
                    pass
                count = count + 1
            return result

        @app.route("/networkgraph")
        def network_graph():
            try:
                plot = Plot(self.control.statestore)
                result = netgraph()
                df_nodes = pd.DataFrame(result["nodes"])
                df_edges = pd.DataFrame(result["edges"])
                graph = plot.make_netgraph_plot(df_edges, df_nodes)
                return json.dumps(json_item(graph, "myplot"))
            except Exception:
                raise
                # return ''

        @app.route("/events")
        def events():
            """

            :return:
            """

            response = self.control.get_events()
            events = []

            result = response["result"]

            for evt in result:
                events.append(evt)

            return jsonify({"result": events, "count": response["count"]})

            json_docs = []
            for doc in self.control.get_events():
                json_doc = json.dumps(doc, default=json_util.default)
                json_docs.append(json_doc)

            json_docs.reverse()

            return {"events": json_docs}

        @app.route("/add")
        def add():
            """Add a combiner to the network."""
            print("Adding combiner to network:", flush=True)
            if self.token_auth_enabled:
                self.authorize(request, app.config.get("SECRET_KEY"))
            if self.control.state() == ReducerState.setup:
                return jsonify({"status": "retry"})

            name = request.args.get("name", None)
            address = str(request.args.get("address", None))
            fqdn = str(request.args.get("fqdn", None))
            port = request.args.get("port", None)
            secure_grpc = request.args.get("secure", None)

            if (
                port is None
                or address is None
                or name is None
                or secure_grpc is None
            ):
                return "Please specify correct parameters."

            # Try to retrieve combiner from db
            combiner = self.control.network.get_combiner(name)
            if not combiner:
                if secure_grpc == "True":
                    certificate, key = self.certificate_manager.get_or_create(
                        address
                    ).get_keypair_raw()
                    _ = base64.b64encode(certificate)
                    _ = base64.b64encode(key)

                else:
                    certificate = None
                    key = None

                combiner = CombinerInterface(
                    self,
                    name=name,
                    address=address,
                    fqdn=fqdn,
                    port=port,
                    certificate=copy.deepcopy(certificate),
                    key=copy.deepcopy(key),
                    ip=request.remote_addr,
                )

                self.control.network.add_combiner(combiner)

            combiner = self.control.network.get_combiner(name)

            ret = {
                "status": "added",
                "storage": self.control.statestore.get_storage_backend(),
                "statestore": self.control.statestore.get_config(),
                "certificate": combiner.get_certificate(),
                "key": combiner.get_key(),
            }

            return jsonify(ret)

        @app.route("/eula", methods=["GET", "POST"])
        def eula():
            """

            :return:
            """
            for r in request.headers:
                print("header contains: {}".format(r), flush=True)

            return render_template("eula.html", configured=True)

        @app.route("/models", methods=["GET", "POST"])
        def models():
            """

            :return:
            """
            # Token auth
            if self.token_auth_enabled:
                self.authorize(request, app.config.get("SECRET_KEY"))

            if request.method == "POST":
                # upload seed file
                uploaded_seed = request.files["seed"]
                if uploaded_seed:
                    a = BytesIO()
                    a.seek(0, 0)
                    uploaded_seed.seek(0)
                    a.write(uploaded_seed.read())
                    helper = self.control.get_helper()
                    a.seek(0)
                    model = helper.load(a)
                    self.control.commit(uploaded_seed.filename, model)
            else:
                not_configured = self.check_configured()
                if not_configured:
                    return not_configured

                plot = Plot(self.control.statestore)
                try:
                    valid_metrics = plot.fetch_valid_metrics()
                    box_plot = plot.create_box_plot(valid_metrics[0])
                except Exception as e:
                    valid_metrics = None
                    box_plot = None
                    print(e, flush=True)

                h_latest_model_id = self.statestore.get_latest_model()

                model_info = self.control.get_model_info()
                return render_template(
                    "models.html",
                    box_plot=box_plot,
                    metrics=valid_metrics,
                    h_latest_model_id=h_latest_model_id,
                    seed=True,
                    model_info=model_info,
                    configured=True,
                )

            seed = True
            return redirect(url_for("models", seed=seed))

        @app.route("/delete_model_trail", methods=["GET", "POST"])
        def delete_model_trail():
            """

            :return:
            """
            if request.method == "POST":
                statestore_config = self.control.statestore.get_config()
                self.tracer = MongoTracer(
                    statestore_config["mongo_config"],
                    statestore_config["network_id"],
                )
                try:
                    self.control.drop_models()
                except Exception:
                    pass

                # drop objects in minio
                self.control.delete_bucket_objects()
                return redirect(url_for("models"))
            seed = True
            return redirect(url_for("models", seed=seed))

        @app.route("/drop_control", methods=["GET", "POST"])
        def drop_control():
            """

            :return:
            """
            if request.method == "POST":
                self.control.statestore.drop_control()
                return redirect(url_for("control"))
            return redirect(url_for("control"))

        # http://localhost:8090/control?rounds=4&model_id=879fa112-c861-4cb1-a25d-775153e5b548
        @app.route("/control", methods=["GET", "POST"])
        def control():
            """Main page for round control. Configure, start and stop training sessions."""
            # Token auth
            if self.token_auth_enabled:
                self.authorize(request, app.config.get("SECRET_KEY"))

            not_configured = self.check_configured()
            if not_configured:
                return not_configured

            state = ReducerStateToString(self.control.state())
            refresh = True

            if self.remote_compute_package:
                try:
                    self.current_compute_context = (
                        self.control.get_compute_package_name()
                    )
                except Exception:
                    self.current_compute_context = None
            else:
                self.current_compute_context = "None:Local"
            if self.control.state() == ReducerState.monitoring:
                return redirect(
                    url_for(
                        "index",
                        state=state,
                        refresh=refresh,
                        message="Reducer is in monitoring state",
                    )
                )

            if request.method == "POST":
                # Get session configuration
                round_timeout = float(request.form.get("timeout", 180))
                buffer_size = int(request.form.get("buffer_size", -1))
                rounds = int(request.form.get("rounds", 1))
                delete_models = request.form.get("delete_models", True)
                task = request.form.get("task", "")
                clients_required = request.form.get("clients_required", 1)
                clients_requested = request.form.get("clients_requested", 8)

                # checking if there are enough clients connected to start!
                clients_available = 0
                for combiner in self.control.network.get_combiners():
                    try:
                        combiner_state = combiner.report()
                        nac = combiner_state["nr_active_clients"]
                        clients_available = clients_available + int(nac)
                    except Exception:
                        pass

                if clients_available < clients_required:
                    return redirect(
                        url_for(
                            "index",
                            state=state,
                            message="Not enough clients available to start rounds! "
                            "check combiner client capacity",
                            message_type="warning",
                        )
                    )

                validate = request.form.get("validate", False)
                if validate == "False":
                    validate = False
                helper_type = request.form.get("helper", "keras")
                # self.control.statestore.set_framework(helper_type)

                latest_model_id = self.statestore.get_latest_model()

                config = {
                    "round_timeout": round_timeout,
                    "buffer_size": buffer_size,
                    "model_id": latest_model_id,
                    "rounds": rounds,
                    "delete_models_storage": delete_models,
                    "clients_required": clients_required,
                    "clients_requested": clients_requested,
                    "task": task,
                    "validate": validate,
                    "helper_type": helper_type,
                }

                threading.Thread(
                    target=self.control.session, args=(config,)
                ).start()

                return redirect(
                    url_for(
                        "index",
                        state=state,
                        refresh=refresh,
                        message="Sent execution plan.",
                        message_type="SUCCESS",
                    )
                )

            else:
                seed_model_id = None
                latest_model_id = None
                try:
                    seed_model_id = self.statestore.get_initial_model()
                    latest_model_id = self.statestore.get_latest_model()
                except Exception:
                    pass

                return render_template(
                    "index.html",
                    latest_model_id=latest_model_id,
                    compute_package=self.current_compute_context,
                    seed_model_id=seed_model_id,
                    helper=self.control.statestore.get_helper(),
                    validate=True,
                    configured=True,
                )

        @app.route("/assign")
        def assign():
            """Handle client assignment requests."""

            if self.token_auth_enabled:
                self.authorize(request, app.config.get("SECRET_KEY"))

            response = self.check_configured_response()

            if response:
                return response

            name = request.args.get("name", None)
            combiner_preferred = request.args.get("combiner", None)

            if combiner_preferred:
                combiner = self.control.network.get_combiner(
                    combiner_preferred
                )
            else:
                combiner = self.control.network.find_available_combiner()

            if combiner is None:
                return jsonify(
                    {
                        "status": "retry",
                        "package": self.package,
                        "msg": "Failed to assign to a combiner, try again later.",
                    }
                )

            client = {
                "name": name,
                "combiner_preferred": combiner_preferred,
                "combiner": combiner.name,
                "ip": request.remote_addr,
                "status": "available",
            }

            # Add client to database
            self.control.network.add_client(client)

            # Return connection information to client
            if combiner.certificate:
                cert_b64 = base64.b64encode(combiner.certificate)
                cert = str(cert_b64).split("'")[1]
            else:
                cert = None

            response = {
                "status": "assigned",
                "host": combiner.address,
                "fqdn": combiner.fqdn,
                "package": self.package,
                "ip": combiner.ip,
                "port": combiner.port,
                "certificate": cert,
                "model_type": self.control.statestore.get_helper(),
            }

            return jsonify(response)

        def combiner_status():
            """Get current status reports from all combiners registered in the network.

            :return:
            """
            combiner_info = []
            for combiner in self.control.network.get_combiners():
                try:
                    report = combiner.report()
                    combiner_info.append(report)
                except Exception:
                    pass
            return combiner_info

        def client_status():
            """
            Get current status of clients (available) from DB compared with client status from all combiners,
            update client status to DB and add their roles.
            """
            client_info = self.control.network.get_client_info()
            combiner_info = combiner_status()
            try:
                all_active_trainers = []
                all_active_validators = []

                for client in combiner_info:
                    active_trainers_str = client["active_trainers"]
                    active_validators_str = client["active_validators"]
                    active_trainers_str = re.sub(
                        "[^a-zA-Z0-9-:\n\.]", "", active_trainers_str
                    ).replace(
                        "name:", " "
                    )  # noqa: W605
                    active_validators_str = re.sub(
                        "[^a-zA-Z0-9-:\n\.]", "", active_validators_str
                    ).replace(
                        "name:", " "
                    )  # noqa: W605
                    all_active_trainers.extend(
                        " ".join(active_trainers_str.split(" ")).split()
                    )
                    all_active_validators.extend(
                        " ".join(active_validators_str.split(" ")).split()
                    )

                active_trainers_list = [
                    client
                    for client in client_info
                    if client["name"] in all_active_trainers
                ]
                active_validators_list = [
                    cl
                    for cl in client_info
                    if cl["name"] in all_active_validators
                ]
                all_clients = [cl for cl in client_info]

                for client in all_clients:
                    status = "offline"
                    role = "None"
                    self.control.network.update_client_data(
                        client, status, role
                    )

                all_active_clients = (
                    active_validators_list + active_trainers_list
                )
                for client in all_active_clients:
                    status = "active"
                    if (
                        client in active_trainers_list
                        and client in active_validators_list
                    ):
                        role = "trainer-validator"
                    elif client in active_trainers_list:
                        role = "trainer"
                    elif client in active_validators_list:
                        role = "validator"
                    else:
                        role = "unknown"
                    self.control.network.update_client_data(
                        client, status, role
                    )

                return {
                    "active_clients": all_clients,
                    "active_trainers": active_trainers_list,
                    "active_validators": active_validators_list,
                }
            except Exception:
                pass

            return {
                "active_clients": [],
                "active_trainers": [],
                "active_validators": [],
            }

        @app.route("/metric_type", methods=["GET", "POST"])
        def change_features():
            """

            :return:
            """
            feature = request.args["selected"]
            plot = Plot(self.control.statestore)
            graphJSON = plot.create_box_plot(feature)
            return graphJSON

        @app.route("/dashboard")
        def dashboard():
            """

            :return:
            """
            # Token auth
            if self.token_auth_enabled:
                self.authorize(request, app.config.get("SECRET_KEY"))

            not_configured = self.check_configured()
            if not_configured:
                return not_configured

            plot = Plot(self.control.statestore)
            combiners_plot = plot.create_combiner_plot()

            timeline_plot = None
            table_plot = None
            clients_plot = plot.create_client_plot()
            client_histogram_plot = plot.create_client_histogram_plot()

            return render_template(
                "dashboard.html",
                show_plot=True,
                table_plot=table_plot,
                timeline_plot=timeline_plot,
                clients_plot=clients_plot,
                client_histogram_plot=client_histogram_plot,
                combiners_plot=combiners_plot,
                configured=True,
            )

        @app.route("/network")
        def network():
            """

            :return:
            """
            # Token auth
            if self.token_auth_enabled:
                self.authorize(request, app.config.get("SECRET_KEY"))

            not_configured = self.check_configured()
            if not_configured:
                return not_configured
            plot = Plot(self.control.statestore)
            round_time_plot = plot.create_round_plot()
            mem_cpu_plot = plot.create_cpu_plot()
            combiner_info = combiner_status()
            active_clients = client_status()
            # print(combiner_info, flush=True)
            return render_template(
                "network.html",
                network_plot=True,
                round_time_plot=round_time_plot,
                mem_cpu_plot=mem_cpu_plot,
                combiner_info=combiner_info,
                active_clients=active_clients["active_clients"],
                active_trainers=active_clients["active_trainers"],
                active_validators=active_clients["active_validators"],
                configured=True,
            )

        @app.route("/config/download", methods=["GET"])
        def config_download():
            """

            :return:
            """
            chk_string = ""
            name = self.control.get_compute_package_name()
            if name is None or name == "":
                chk_string = ""
            else:
                file_path = os.path.join(UPLOAD_FOLDER, name)
                print("trying to get {}".format(file_path))

                try:
                    sum = str(sha(file_path))
                except FileNotFoundError:
                    sum = ""
                chk_string = "checksum: {}".format(sum)

            network_id = self.network_id
            discover_host = self.name
            discover_port = self.port
            ctx = """network_id: {network_id}
discover_host: {discover_host}
discover_port: {discover_port}
{chk_string}""".format(
                network_id=network_id,
                discover_host=discover_host,
                discover_port=discover_port,
                chk_string=chk_string,
            )

            obj = BytesIO()
            obj.write(ctx.encode("UTF-8"))
            obj.seek(0)
            return send_file(
                obj,
                as_attachment=True,
                download_name="client.yaml",
                mimetype="application/x-yaml",
            )

        @app.route("/context", methods=["GET", "POST"])
        def context():
            """

            :return:
            """
            # Token auth
            if self.token_auth_enabled:
                self.authorize(request, app.config.get("SECRET_KEY"))

            # if reset is not empty then allow context re-set
            reset = request.args.get("reset", None)
            if reset:
                return render_template("context.html")

            if request.method == "POST":
                if "file" not in request.files:
                    flash("No file part")
                    return redirect(url_for("context"))

                file = request.files["file"]
                helper_type = request.form.get("helper", "kerashelper")
                # if user does not select file, browser also
                # submit an empty part without filename
                if file.filename == "":
                    flash("No selected file")
                    return redirect(url_for("context"))

                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(
                        app.config["UPLOAD_FOLDER"], filename
                    )
                    file.save(file_path)

                    if (
                        self.control.state() == ReducerState.instructing
                        or self.control.state() == ReducerState.monitoring
                    ):
                        return "Not allowed to change context while execution is ongoing."

                    self.control.set_compute_package(filename, file_path)
                    self.control.statestore.set_helper(helper_type)
                    return redirect(url_for("control"))

            name = request.args.get("name", "")

            if name == "":
                name = self.control.get_compute_package_name()
                if name is None or name == "":
                    return render_template("context.html")

            # There is a potential race condition here, if one client requests a package and at
            # the same time another one triggers a fetch from Minio and writes to disk.
            try:
                mutex = Lock()
                mutex.acquire()
                return send_from_directory(
                    app.config["UPLOAD_FOLDER"], name, as_attachment=True
                )
            except Exception:
                try:
                    data = self.control.get_compute_package(name)
                    file_path = os.path.join(app.config["UPLOAD_FOLDER"], name)
                    with open(file_path, "wb") as fh:
                        fh.write(data)
                    return send_from_directory(
                        app.config["UPLOAD_FOLDER"], name, as_attachment=True
                    )
                except Exception:
                    raise
            finally:
                mutex.release()

            return render_template("context.html")

        @app.route("/checksum", methods=["GET", "POST"])
        def checksum():
            """

            :return:
            """
            # sum = ''
            name = request.args.get("name", None)
            if name == "" or name is None:
                name = self.control.get_compute_package_name()
                if name is None or name == "":
                    return jsonify({})

            file_path = os.path.join(UPLOAD_FOLDER, name)
            print("trying to get {}".format(file_path))

            try:
                sum = str(sha(file_path))
            except FileNotFoundError:
                sum = ""

            data = {"checksum": sum}

            return jsonify(data)

        @app.route("/infer", methods=["POST"])
        def infer():
            """

            :return:
            """
            # Token auth
            if self.token_auth_enabled:
                self.authorize(request, app.config.get("SECRET_KEY"))

            # Check configured
            not_configured = self.check_configured()
            if not_configured:
                return not_configured

            # Check compute context
            if self.remote_compute_context:
                try:
                    self.current_compute_context = (
                        self.control.get_compute_package()
                    )
                except Exception as e:
                    print(e, flush=True)
                    self.current_compute_context = None
            else:
                self.current_compute_context = "None:Local"

            # Redirect if in monitoring state
            if self.control.state() == ReducerState.monitoring:
                return redirect(
                    url_for(
                        "index",
                        state=ReducerStateToString(self.control.state()),
                        refresh=True,
                        message="Reducer is in monitoring state",
                    )
                )

            # POST params
            timeout = int(request.form.get("timeout", 180))
            helper_type = request.form.get("helper", "keras")
            clients_required = request.form.get("clients_required", 1)
            clients_requested = request.form.get("clients_requested", 8)

            # Start inference request
            config = {
                "round_timeout": timeout,
                "model_id": self.statestore.get_latest_model(),
                "clients_required": clients_required,
                "clients_requested": clients_requested,
                "task": "inference",
                "helper_type": helper_type,
            }
            threading.Thread(
                target=self.control.infer_instruct, args=(config,)
            ).start()

            # Redirect
            return redirect(
                url_for(
                    "index",
                    state=ReducerStateToString(self.control.state()),
                    refresh=True,
                    message="Sent execution plan (inference).",
                    message_type="SUCCESS",
                )
            )

        if not self.host:
            bind = "0.0.0.0"
        else:
            bind = self.host

        app.run(host=bind, port=self.port)

        return app
