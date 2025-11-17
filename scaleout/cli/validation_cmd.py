import click

from scaleout.cli.main import main
from scaleout.cli.shared import complement_with_context, get_response, print_response


@main.group(
    "validation",
    help="Commands to list and get validation results.",
    invoke_without_command=True,
)
@click.pass_context
def validation_cmd(ctx):
    """Validation commands."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-s", "--session_id", required=False, help="validations in session with given session id")
@click.option("--n_max", required=False, help="Number of items to list")
@validation_cmd.command("list")
@click.pass_context
def list_validations(ctx, protocol: str, host: str, port: str, token: str = None, session_id: str = None, n_max: int = None):
    """Return:
    ------
    - count: number of validations
    - result: list of validations

    """
    base_url, token = complement_with_context(protocol, host, port, token)
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    query = {}
    if session_id:
        query["session_id"] = session_id

    response = get_response(base_url=base_url, endpoint="validations/", query=query, token=token, headers=headers)
    print_response(response, "validations")


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-id", "--id", required=True, help="validation ID")
@validation_cmd.command("get")
@click.pass_context
def get_validation(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None):
    """Return:
    ------
    - result: validation with given id

    """
    base_url, token = complement_with_context(protocol, host, port, token)
    response = get_response(base_url=base_url, endpoint=f"validations/{id}", query={}, token=token, headers={})
    print_response(response, "validation")
