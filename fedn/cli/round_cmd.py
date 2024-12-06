import click
import requests

from .main import main
from .shared import CONTROLLER_DEFAULTS, get_api_url, get_token, print_response


@main.group("round")
@click.pass_context
def round_cmd(ctx):
    """:param ctx:
    """
    pass


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-session_id", "--session_id", required=False, help="Rounds in session with given session id")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("--n_max", required=False, help="Number of items to list")
@round_cmd.command("list")
@click.pass_context
def list_rounds(ctx, protocol: str, host: str, port: str, token: str = None, session_id: str = None, n_max: int = None):
    """Return:
    ------
    - count: number of rounds
    - result: list of rounds

    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint="rounds")

    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    _token = get_token(token)

    if _token:
        headers["Authorization"] = _token

    if session_id:
        url = f"{url}?round_config.session_id={session_id}"


    try:
        response = requests.get(url, headers=headers)
        print_response(response, "rounds", None)

    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url}")


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-id", "--id", required=True, help="Round ID")
@click.option("-t", "--token", required=False, help="Authentication token")
@round_cmd.command("get")
@click.pass_context
def get_round(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None):
    """Return:
    ------
    - result: round with given id

    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint="rounds")

    headers = {}


    _token = get_token(token)

    if _token:
        headers["Authorization"] = _token

    if id:
        url = f"{url}{id}"


    try:
        response = requests.get(url, headers=headers)
        print_response(response, "round", id)

    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url}")
