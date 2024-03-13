import json

import click
import requests

from .main import main
from .shared import API_VERSION, CONTROLLER_DEFAULTS


@main.group('start')
@click.pass_context
def start_cmd(ctx):
    """
    - Issue commands to the network.
    """
    pass


@click.option('-h', '--host', required=True, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api).')
@click.option('-i', '--port', required=True, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api).')
@click.option('-n', '--name', required=False, help='Name of session (session id).')
@click.option('-v', '--validate', required=True, default=True, help='Validate the session. Set to False to skip validation.')
@start_cmd.command('session')
@click.pass_context
def start_session(ctx, host: str, port: int, name: str = None, validate: bool = True):
    """
    - Start a session.
    return: None
    """
    url = f'http://{host}:{port}/start_session'
    headers = {'Content-Type': 'application/json'}

    click.echo(f'\nStarting session: {url}\n')

    try:

        data = {'session_id': name, 'validate': validate}
        response = requests.post(url, data=json.dumps(data), headers=headers)

        if response.status_code == 200:
            json_data = response.json()
            click.echo(json.dumps(json_data, indent=4))
        else:
            click.echo("Something went wrong. Please try again.")

    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {host}:{port}')
