import click
from .main import main
from fedn.network.api.server import start_server_api

@main.group("controller")
@click.pass_context
def controller_cmd(ctx):
    """:param ctx:
    """
    pass
@controller_cmd.command("start")
@click.pass_context
def controller_cmd(ctx):
    start_server_api()
