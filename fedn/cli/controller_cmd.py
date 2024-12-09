import click

from fedn.cli.main import main


@main.group("controller")
@click.pass_context
def controller_cmd(ctx):
    """:param ctx:"""
    pass


@controller_cmd.command("start")
@click.pass_context
def controller_cmd(ctx):
    from fedn.network.api.server import start_server_api

    start_server_api()
