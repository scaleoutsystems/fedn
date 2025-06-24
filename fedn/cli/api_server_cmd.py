import click

from fedn.cli.main import main


@main.group("api-server")
@click.pass_context
def api_server_cmd(ctx):
    """:param ctx:"""
    pass


@api_server_cmd.command("start")
@click.pass_context
def api_server_cmd(ctx):
    from fedn.network.api.server import start_api_server  # noqa: PLC0415

    start_api_server()
