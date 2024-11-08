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
@click.option("-id", "--id", required=False, help="Round ID")
@click.option("-session_id", "--session_id", required=False, help="Rounds in session with given session id")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("--n_max", required=False, help="Number of items to list")
@round_cmd.command("list")
@click.pass_context
def list_rounds(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None, session_id: str = None, n_max: int = None):
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

    if id:
        url = f"{url}{id}"
        headers["id"] = id

    click.echo(f"\nListing rounds: {url}\n")
    click.echo(f"Headers: {headers}")
    try:
        response = requests.get(url, headers=headers)
        if session_id:
            if response.status_code == 200:
                json_data = response.json()
                count, result = json_data.values()
                click.echo(f"Found {count} rounds")
                click.echo("\n---------------------------------\n")
                for obj in result:
                    if obj.get("round_config").get("session_id")==session_id:
                        click.echo("{")
                        for k, v in obj.items():
                            click.echo(f"\t{k}: {v}")
                        click.echo("}")

            elif response.status_code == 500:
                json_data = response.json()
                click.echo(f'Error: {json_data["message"]}')
            else:
                click.echo(f"Error: {response.status_code}")
        else:
            if id:
                print_response(response, "round", True, session_id)
            else:
                print_response(response, "rounds", False, session_id)


    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url}")
