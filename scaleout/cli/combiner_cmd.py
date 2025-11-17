import click

from scaleout.cli.main import main
from scaleout.cli.shared import get_response, print_response, complement_with_context


@main.group(
    "combiner",
    help="Commands to list and inspect combiners.",
    invoke_without_command=True,
)
@click.pass_context
def combiner_cmd(ctx):
    """Combiner commands."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("--n_max", required=False, help="Number of items to list")
@combiner_cmd.command("list")
@click.pass_context
def list_combiners(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None):
    """Return:
    ------
    - count: number of combiners
    - result: list of combiners

    """
    base_url, token = complement_with_context(protocol, host, port, token)
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    response = get_response(base_url=base_url, endpoint="combiners/", query={}, token=token, headers=headers)
    print_response(response, "combiners")


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-id", "--id", required=True, help="Combiner ID")
@combiner_cmd.command("get")
@click.pass_context
def get_combiner(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None):
    """Return:
    ------
    - result: combiner with given id

    """
    base_url, token = complement_with_context(protocol, host, port, token)
    response = get_response(base_url=base_url, endpoint=f"combiners/{id}", query={}, token=token, headers={})
    print_response(response, "combiner")
