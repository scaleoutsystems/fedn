import click

from fedn.network.combiner.hooks.hooks import serve

from .main import main


@main.group("hooks")
@click.pass_context
def hooks_cmd(ctx):
    """:param ctx:"""
    pass


@hooks_cmd.command("start")
@click.pass_context
def start_cmd(ctx):
    """:param ctx:"""
    click.echo("Started hooks container")
    serve()
