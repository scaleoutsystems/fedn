import click

from .main import main
from .shared import CONTROLLER_DEFAULTS, get_response, print_response


@main.group("status")
@click.pass_context
def status_cmd(ctx):
    """:param ctx:"""
    pass


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-s", "--session_id", required=False, help="statuses with given session id")
@click.option("--n_max", required=False, help="Number of items to list")
@status_cmd.command("list")
@click.pass_context
def list_statuses(ctx, protocol: str, host: str, port: str, token: str = None, session_id: str = None, n_max: int = None):
    """Return:
    ------
    - count: number of statuses
    - result: list of statuses

    """
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    if session_id:
        response = get_response(
            protocol=protocol, host=host, port=port, endpoint=f"statuses/?sessionId={session_id}", token=token, headers=headers, usr_api=False, usr_token=False
        )
    else:
        response = get_response(protocol=protocol, host=host, port=port, endpoint="statuses/", token=token, headers=headers, usr_api=False, usr_token=False)
    print_response(response, "statuses", None)


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
    response = get_response(protocol=protocol, host=host, port=port, endpoint=f"statuses{id}", token=token, headers={}, usr_api=False, usr_token=False)
    print_response(response, "status", id)
