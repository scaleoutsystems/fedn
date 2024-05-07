import click
import requests

from .main import main
from .shared import CONTROLLER_DEFAULTS, get_api_url, get_token, print_response


@main.group("status")
@click.pass_context
def status_cmd(ctx):
    """

    :param ctx:
    """
    pass


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("--n_max", required=False, help="Number of items to list")
@status_cmd.command("list")
@click.pass_context
def list_statuses(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None):
    """
    Return:
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

    click.echo(f"\nListing statuses: {url}\n")
    click.echo(f"Headers: {headers}")

    try:
        response = requests.get(url, headers=headers)
        print_response(response, "statuses")
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url}")
