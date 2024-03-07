import click
import requests

from .main import main
from .shared import API_VERSION, CONTROLLER_DEFAULTS


@main.group('upload')
@click.pass_context
def upload_cmd(ctx):
    """
    :param ctx:
    """
    pass


@click.option('-h', '--host', required=True, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api).')
@click.option('-i', '--port', required=True, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api).')
@click.option('-hl', '--helper', required=True, default="numpyhelper", help='Helper type to use.')
@click.option('-p', '--path', required=True, help='Path to package file.')
@click.option('-n', '--name', required=False, help='Name of package.')
@click.option('-d', '--description', required=False, help='Description of package.')
@upload_cmd.command('package')
@click.pass_context
def upload_package(ctx, host: str, port: int, helper: str, path: str, name: str = None, description: str = None):
    """
    - Upload a package to the controller.
    return: None
    """
    url = f'http://{host}:{port}/set_package'
    headers = {}

    click.echo(f'\nUploading package: {url}\n')

    try:
        with open(path, 'rb') as file:
            response = requests.post(url, files={'file': file}, data={
                                     'helper': helper, 'name': name, 'description': description})

            if response.status_code == 200:
                click.echo("Success! Package uploaded.")
            else: 
                click.echo("Something went wrong. Please try again.")
    except FileNotFoundError:
        click.echo(f'Error: Could not find file {path}')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {host}:{port}')


@click.option('-h', '--host', required=True, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api).')
@click.option('-i', '--port', required=True, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api).')
@click.option('-hl', '--helper', required=True, default="numpyhelper", help='Helper type to use.')
@click.option('-p', '--path', required=True, help='Path to model file.')
@click.option('-n', '--name', required=False, help='Name of model.')
@click.option('-d', '--description', required=False, help='Description of model.')
@upload_cmd.command('model')
@click.pass_context
def upload_model(ctx, host: str, port: int, helper: str, path: str, name: str = None, description: str = None):
    """
    - Upload a model to the controller.
    return: None
    """
    url = f'http://{host}:{port}/set_initial_model'
    headers = {}

    click.echo(f'\nUploading model: {url}\n')

    try:
        with open(path, 'rb') as file:
            response = requests.post(url, files={'file': file}, data={
                                     'helper': helper, 'name': name, 'description': description})

            if response.status_code == 200:
                click.echo("Success! model uploaded.")
            else: 
                click.echo("Something went wrong. Please try again.")
    except FileNotFoundError:
        click.echo(f'Error: Could not find file {path}')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {host}:{port}')
