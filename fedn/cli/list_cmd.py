
import click

from .main import main


@main.group('list')
@click.pass_context
def list_cmd(ctx):
    """
    :param ctx:
    """
    pass


@list_cmd.command('models')
def list_models():
    """
    :return:
    """
    click.echo('Listing models')
