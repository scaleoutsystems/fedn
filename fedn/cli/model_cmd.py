import click

from .main import main
from .shared import CONTROLLER_DEFAULTS, get_response, print_response


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
    """Return:
    ------
    - count: number of models
    - result: list of models

    """
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    if session_id:
        response = get_response(
            protocol=protocol, host=host, port=port, endpoint=f"models/?session_id={session_id}", token=token, headers=headers, usr_api=False, usr_token=False
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
    """Return:
    ------
    - result: model with given id

    """
    response = get_response(protocol=protocol, host=host, port=port, endpoint=f"models/{id}", token=token, headers={}, usr_api=False, usr_token=False)
    print_response(response, "model", id)
