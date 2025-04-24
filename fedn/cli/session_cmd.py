import click
import requests

from .main import main
from .shared import CONTROLLER_DEFAULTS, get_api_url, get_response, get_token, print_response


@main.group("session")
@click.pass_context
def session_cmd(ctx):
    """Session commands."""
    pass


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("--n_max", required=False, help="Number of items to list")
@session_cmd.command("list")
@click.pass_context
def list_sessions(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None):
    """List sessions."""
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    response = get_response(protocol=protocol, host=host, port=port, endpoint="sessions/", token=token, headers=headers, usr_api=False, usr_token=False)
    print_response(response, "sessions", None)


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-id", "--id", required=True, help="Session ID")
@session_cmd.command("get")
@click.pass_context
def get_session(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None):
    """Get session by id."""
    response = get_response(protocol=protocol, host=host, port=port, endpoint=f"sessions/{id}", token=token, headers={}, usr_api=False, usr_token=False)
    print_response(response, "session", id)


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-n", "--name", required=False, help="Name of the session")
@click.option("-a", "--aggregator", required=False, default="fedavg", help="The aggregator plugin to use")
@click.option("-ak", "--aggregator_kwargs", required=False, type=dict, help="Aggregator keyword arguments")
@click.option("-m", "--model_id", required=False, help="The id of the initial model")
@click.option("-rt", "--round_timeout", required=False, default=180, type=int, help="The round timeout to use in seconds")
@click.option("-r", "--rounds", required=False, default=5, type=int, help="The number of rounds to perform")
@click.option("-rb", "--round_buffer_size", required=False, default=-1, type=int, help="The round buffer size to use")
@click.option("-d", "--delete_models", required=False, default=True, type=bool, help="Whether to delete models after each round at combiner (save storage)")
@click.option("-v", "--validate", required=False, default=True, type=bool, help="Whether to validate the model after each round")
@click.option("-hp", "--helper", required=False, help="The helper type to use")
@click.option("-mc", "--min_clients", required=False, default=1, type=int, help="The minimum number of clients required")
@click.option("-rc", "--requested_clients", required=False, default=8, type=int, help="The requested number of clients")
@session_cmd.command("start")
@click.pass_context
def start_session(
    ctx,
    protocol: str,
    host: str,
    port: str,
    token: str,
    name: str = None,
    aggregator: str = "fedavg",
    aggregator_kwargs: dict = None,
    model_id: str = None,
    round_timeout: int = 180,
    rounds: int = 5,
    round_buffer_size: int = -1,
    delete_models: bool = True,
    validate: bool = True,
    helper: str = None,
    min_clients: int = 1,
    requested_clients: int = 8,
):
    """Start a new session."""
    headers = {}
    _token = get_token(token=token, usr_token=False)
    if _token:
        headers = {"Authorization": _token}

    if model_id is None:
        model_query_headers = headers.copy()
        model_query_headers["X-Limit"] = "1"
        model_query_headers["X-Sort-Key"] = "committed_at"
        model_query_headers["X-Sort-Order"] = "desc"

        url = get_api_url(protocol, host, port, "models", usr_api=False)
        response = requests.get(url, headers=model_query_headers)
        if response.status_code == 200:
            json = response.json()

            if "result" in json and len(json["result"]) > 0:
                model_id = json["result"][0]["model_id"]
            else:
                click.secho("No models found", fg="red")
                return
        else:
            click.secho(f"Failed to get active model: {response.json()}", fg="red")
            return

    if helper is None:
        url = get_api_url(protocol, host, port, "helpers/active", usr_api=False)
        response = requests.get(url, headers=headers)
        if response.status_code == 400:
            helper = "numpyhelper"
        elif response.status_code == 200:
            helper = response.json()
        else:
            click.secho("An unexpected error occurred when getting the active helper", fg="red")
            return

    url = get_api_url(protocol, host, port, "sessions/", usr_api=False)
    response = requests.post(
        url,
        json={
            "name": name,
            "session_config": {
                "aggregator": aggregator,
                "aggregator_kwargs": aggregator_kwargs,
                "round_timeout": round_timeout,
                "buffer_size": round_buffer_size,
                "model_id": model_id,
                "delete_models_storage": delete_models,
                "clients_required": min_clients,
                "requested_clients": requested_clients,
                "validate": validate,
                "helper_type": helper,
                "server_functions": None,
            },
        },
        headers=headers,
        verify=False,
    )

    if response.status_code == 201:
        session_id = response.json()["session_id"]
        url = get_api_url(protocol, host, port, "sessions/start", usr_api=False)
        try:
            response = requests.post(
                url,
                json={
                    "session_id": session_id,
                    "rounds": rounds,
                    "round_timeout": round_timeout,
                },
                headers=headers,
                verify=False,
            )
            response_json = response.json()
            response_json["session_id"] = session_id
            click.secho(f"Session started successfully: {response_json}", fg="green")
        except requests.exceptions.RequestException:
            click.secho(f"Failed to start session: {response.json()}", fg="red")
    else:
        click.secho(f"Failed to start session: {response.json()}", fg="red")
