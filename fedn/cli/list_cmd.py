

import click
import requests

from .main import main
from .shared import API_VERSION, CONTROLLER_DEFAULTS


def print_response(response, entity_name: str):
    """
    :param response:
    :return:
    """
    if response.status_code == 200:
        json_data = response.json()
        count, result = json_data.values()
        click.echo(f'Found {count} {entity_name}')
        click.echo('\n---------------------------------\n')
        for obj in result:
            click.echo('{')
            for k, v in obj.items():
                click.echo(f'\t{k}: {v}')
            click.echo('}')
    elif response.status_code == 500:
        json_data = response.json()
        click.echo(f'Error: {json_data["message"]}')
    else:
        click.echo(f'Error: {response.status_code}')

@main.group('list')
@click.pass_context
def list_cmd(ctx):
    """
    :param ctx:
    """
    pass


@click.option('-h', '--host', required=False, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api).')
@click.option('-i', '--port', required=False, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api).')
@click.option('--n_max', required=False, help='Number of items to list.')
@list_cmd.command('models')
@click.pass_context
def list_models(ctx, host, port, n_max):
    """
    :return:
    """
    url = f'http://{host}:{port}/api/{API_VERSION}/models'
    headers = {}

    if n_max:
        headers['X-Limit'] = n_max

    click.echo(f'\nListing models: {url}\n')

    response = requests.get(url, headers=headers)

    print_response(response, 'models')
