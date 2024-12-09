import click
import requests

from .main import main
from .shared import CONTROLLER_DEFAULTS, get_api_url, get_token, print_response


@main.group("status")
@click.pass_context
def status_cmd(ctx):
    """:param ctx:"""
    pass


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-session_id", "--session_id", required=False, help="statuses with given session id")
@click.option("--n_max", required=False, help="Number of items to list")
@status_cmd.command("list")
@click.pass_context
def list_statuses(ctx, protocol: str, host: str, port: str, token: str = None, session_id: str = None, n_max: int = None):
    """Return:
    ------
    - count: number of statuses
    - result: list of statuses

    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint="statuses")
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    _token = get_token(token)

    if _token:
        headers["Authorization"] = _token

    if session_id:
        url = f"{url}?sessionId={session_id}"


    try:
        response = requests.get(url, headers=headers)
        print_response(response, "statuses", None)
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url}")


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-id", "--id", required=True, help="Status ID")
@status_cmd.command("get")
@click.pass_context
def get_status(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None):
    """Return:
    ------
    - result: status with given id

    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint="statuses")
    headers = {}


    _token = get_token(token)

    if _token:
        headers["Authorization"] = _token

    if id:
        url = f"{url}{id}"


    try:
        response = requests.get(url, headers=headers)
        print_response(response, "status", id)
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url}")
