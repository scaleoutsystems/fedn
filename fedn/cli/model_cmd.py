import click
import requests

from .main import main
from .shared import CONTROLLER_DEFAULTS, get_api_url, get_token, print_response


@main.group("model")
@click.pass_context
def model_cmd(ctx):
    """:param ctx:
    """
    pass


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-session_id", "--session_id", required=False, help="models in session with given session id")
@click.option("--n_max", required=False, help="Number of items to list")
@model_cmd.command("list")
@click.pass_context
def list_models(ctx, protocol: str, host: str, port: str, token: str = None, session_id: str = None, n_max: int = None):
    """Return:
    ------
    - count: number of models
    - result: list of models

    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint="models")


    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    _token = get_token(token)

    if _token:
        headers["Authorization"] = _token

    if session_id:
        url = f"{url}?session_id={session_id}"


    try:
        response = requests.get(url, headers=headers)
        print_response(response, "models", None)
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url}")


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-id", "--id", required=True, help="Model ID")
@model_cmd.command("get")
@click.pass_context
def get_model(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None):
    """Return:
    ------
    - result: model with given id

    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint="models")


    headers = {}


    _token = get_token(token)

    if _token:
        headers["Authorization"] = _token

    if id:
        url = f"{url}{id}"


    try:
        response = requests.get(url, headers=headers)
        print_response(response, "model", id)
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url}")
