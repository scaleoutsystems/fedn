import click
import requests

from .main import main
from .shared import CONTROLLER_DEFAULTS, get_api_url, get_response, get_token, print_response


@main.group("model")
@click.pass_context
def model_cmd(ctx):
    """:param ctx:"""
    pass


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-s", "--session_id", required=False, help="models in session with given session id")
@click.option("--n_max", required=False, help="Number of items to list")
@model_cmd.command("list")
@click.pass_context
def list_models(ctx, protocol: str, host: str, port: str, token: str = None, session_id: str = None, n_max: int = None):
    """List models."""
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    if session_id:
        response = get_response(
            protocol=protocol, host=host, port=port, endpoint=f"models/?session_id={session_id}/", token=token, headers=headers, usr_api=False, usr_token=False
        )
    else:
        response = get_response(protocol=protocol, host=host, port=port, endpoint="models/", token=token, headers=headers, usr_api=False, usr_token=False)
    print_response(response, "models", None)


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-id", "--id", required=True, help="Model ID")
@model_cmd.command("get")
@click.pass_context
def get_model(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None):
    """Get model by id."""
    response = get_response(protocol=protocol, host=host, port=port, endpoint=f"models/{id}", token=token, headers={}, usr_api=False, usr_token=False)
    print_response(response, "model", id)


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-f", "--file", required=True, help="Path to the model file")
@model_cmd.command("set-active")
@click.pass_context
def set_active_model(ctx, protocol: str, host: str, port: str, token: str, file: str):
    """Set the initial model and upload to model repository."""
    headers = {}
    _token = get_token(token=token, usr_token=False)
    if _token:
        headers = {"Authorization": _token}

    if file.endswith(".npz"):
        helper = "numpyhelper"
    elif file.endswith(".bin"):
        helper = "binaryhelper"
    else:
        click.secho("Unsupported file type. Only .npz and .bin files are supported.", fg="red")
        return

    try:
        # Set the active helper
        url = get_api_url(protocol, host, port, "helpers/active", usr_api=False)
        response = requests.put(url, json={"helper": helper}, headers=headers, verify=False)
        response.raise_for_status()

        # Upload the model file
        url = get_api_url(protocol, host, port, "models/", usr_api=False)
        with open(file, "rb") as model_file:
            response = requests.post(url, files={"file": model_file}, data={"helper": helper}, headers=headers, verify=False)
            response.raise_for_status()

        click.secho("Model set as active and uploaded successfully.", fg="green")
    except requests.exceptions.RequestException as e:
        click.secho(f"Failed to set active model: {e}", fg="red")
