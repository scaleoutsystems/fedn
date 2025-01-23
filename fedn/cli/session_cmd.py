import click

from .main import main
from .shared import CONTROLLER_DEFAULTS, get_response, print_response


@main.group("session")
@click.pass_context
def session_cmd(ctx):
    """:param ctx:"""
    pass


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("--n_max", required=False, help="Number of items to list")
@session_cmd.command("list")
@click.pass_context
def list_sessions(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None):
    """Return:
    ------
    - count: number of sessions
    - result: list of sessions

    """
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
    """Return:
    ------
    - result: session with given session id

    """
    response = get_response(protocol=protocol, host=host, port=port, endpoint=f"sessions/{id}", token=token, headers={}, usr_api=False, usr_token=False)
    print_response(response, "session", id)
