import click

from .main import main
from .shared import CONTROLLER_DEFAULTS, get_response, print_response


@main.group("round")
@click.pass_context
def round_cmd(ctx):
    """:param ctx:"""
    pass


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-s", "--session_id", required=False, help="Rounds in session with given session id")
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
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    if session_id:
        response = get_response(
            protocol=protocol,
            host=host,
            port=port,
            endpoint=f"rounds/?round_config.session_id={session_id}",
            token=token,
            headers=headers,
            usr_api=False,
            usr_token=False,
        )
    else:
        response = get_response(
            protocol=protocol,
            host=host,
            port=port,
            endpoint="rounds/",
            token=token,
            headers=headers,
            usr_api=False,
            usr_token=False,
        )
    print_response(response, "rounds", None)


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
    response = get_response(protocol=protocol, host=host, port=port, endpoint=f"rounds/{id}", token=token, headers={}, usr_api=False, usr_token=False)
    print_response(response, "round", id)
