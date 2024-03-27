import os

import click
import requests

from .main import main
from .shared import CONTROLLER_DEFAULTS, get_token


@main.group('upload')
@click.pass_context
def upload_cmd(ctx):
    """
    - Commands used to upload files to the controller.
    """
    pass


def get_api_url(protocol: str, host: str, port: str, endpoint: str) -> str:
    _url = os.environ.get('FEDN_CONTROLLER_URL')

    if _url:
        return f'{_url}/{endpoint}'

    _protocol = os.environ.get('FEDN_PROTOCOL') or protocol or CONTROLLER_DEFAULTS['protocol']
    _host = os.environ.get('FEDN_HOST') or host or CONTROLLER_DEFAULTS['host']
    _port = os.environ.get('FEDN_PORT') or port or CONTROLLER_DEFAULTS['port']

    return f'{_protocol}://{_host}:{_port}/{endpoint}'


@click.option('-p', '--protocol', required=False, default=CONTROLLER_DEFAULTS['protocol'], help='Communication protocol of controller (api)')
@click.option('-H', '--host', required=True, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api)')
@click.option('-P', '--port', required=True, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api)')
@click.option('-t', '--token', required=False, help='Authentication token')
@click.option('-h', '--helper', required=True, default="numpyhelper", help='Helper type to use')
@click.option('-path', '--path', required=True, help='Path to package file')
@click.option('-n', '--name', required=False, help='Name of package')
@click.option('-d', '--description', required=False, help='Description of package')
@upload_cmd.command('package')
@click.pass_context
def upload_package(ctx, protocol: str, host: str, port: int, token: str, helper: str, path: str, name: str = None, description: str = None):
    """
    - Upload a package to the controller.
    return: None
    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint="set_package")
    headers = {}

    _token = get_token(token)

    if _token is not None:
        headers['Authorization'] = _token

    click.echo(f'\nUploading package: {url}\n')
    click.echo(f'Headers: {headers}')

    try:
        with open(path, 'rb') as file:
            response = requests.post(url, files={'file': file}, data={
                                     'helper': helper, 'name': name, 'description': description}, headers=headers)

            if response.status_code == 200:
                click.echo("Success! Package uploaded.")
            else:
                click.echo("Something went wrong. Please try again.")
    except FileNotFoundError:
        click.echo(f'Error: Could not find file {path}')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {url}')


@click.option('-p', '--protocol', required=False, default=CONTROLLER_DEFAULTS['protocol'], help='Communication protocol of controller (api)')
@click.option('-H', '--host', required=True, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api).')
@click.option('-P', '--port', required=True, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api).')
@click.option('-t', '--token', required=False, help='Authentication token')
@click.option('-path', '--path', required=True, help='Path to model file.')
@upload_cmd.command('model')
@click.pass_context
def upload_model(ctx, protocol: str, host: str, port: int, token: str, path: str):
    """
    - Upload a model to the controller.
    return: None
    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint="set_initial_model")

    headers = {}

    _token = get_token(token)

    if _token is not None:
        headers['Authorization'] = _token

    click.echo(f'\nUploading model: {url}\n')
    click.echo(f'Headers: {headers}')

    try:
        with open(path, 'rb') as file:
            response = requests.post(url, files={'file': file}, headers=headers)

            if response.status_code == 200:
                click.echo("Success! Model uploaded.")
            else:
                click.echo("Something went wrong. Please try again.")
    except FileNotFoundError:
        click.echo(f'Error: Could not find file {path}')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {url}')
