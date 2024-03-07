

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
@list_cmd.command('clients')
@click.pass_context
def list_clients(ctx, host, port, n_max):
    """
    return:
    - count: number of clients
    - result: list of clients
    """
    url = f'http://{host}:{port}/api/{API_VERSION}/clients'
    headers = {}

    if n_max:
        headers['X-Limit'] = n_max

    click.echo(f'\nListing clients: {url}\n')

    try:
        response = requests.get(url, headers=headers)
        print_response(response, 'clients')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {host}:{port}')


@click.option('-h', '--host', required=False, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api).')
@click.option('-i', '--port', required=False, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api).')
@click.option('--n_max', required=False, help='Number of items to list.')
@list_cmd.command('combiners')
@click.pass_context
def list_combiners(ctx, host, port, n_max):
    """
    return:
    - count: number of combiners
    - result: list of combiners
    """
    url = f'http://{host}:{port}/api/{API_VERSION}/combiners'
    headers = {}

    if n_max:
        headers['X-Limit'] = n_max

    click.echo(f'\nListing combiners: {url}\n')

    try:
        response = requests.get(url, headers=headers)
        print_response(response, 'combiners')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {host}:{port}')


@click.option('-h', '--host', required=False, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api).')
@click.option('-i', '--port', required=False, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api).')
@click.option('--n_max', required=False, help='Number of items to list.')
@list_cmd.command('models')
@click.pass_context
def list_models(ctx, host, port, n_max):
    """
    return:
    - count: number of models
    - result: list of models
    """
    url = f'http://{host}:{port}/api/{API_VERSION}/models'
    headers = {}

    if n_max:
        headers['X-Limit'] = n_max

    click.echo(f'\nListing models: {url}\n')

    try:
        response = requests.get(url, headers=headers)
        print_response(response, 'models')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {host}:{port}')


@click.option('-h', '--host', required=False, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api).')
@click.option('-i', '--port', required=False, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api).')
@click.option('--n_max', required=False, help='Number of items to list.')
@list_cmd.command('packages')
@click.pass_context
def list_packages(ctx, host, port, n_max):
    """
    return:
    - count: number of packages
    - result: list of packages
    """
    url = f'http://{host}:{port}/api/{API_VERSION}/packages'
    headers = {}

    if n_max:
        headers['X-Limit'] = n_max

    click.echo(f'\nListing packages: {url}\n')

    try:
        response = requests.get(url, headers=headers)
        print_response(response, 'packages')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {host}:{port}')


@click.option('-h', '--host', required=False, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api).')
@click.option('-i', '--port', required=False, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api).')
@click.option('--n_max', required=False, help='Number of items to list.')
@list_cmd.command('rounds')
@click.pass_context
def list_rounds(ctx, host, port, n_max):
    """
    return:
    - count: number of rounds
    - result: list of rounds
    """
    url = f'http://{host}:{port}/api/{API_VERSION}/rounds'
    headers = {}

    if n_max:
        headers['X-Limit'] = n_max

    click.echo(f'\nListing rounds: {url}\n')

    try:
        response = requests.get(url, headers=headers)
        print_response(response, 'rounds')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {host}:{port}')


@click.option('-h', '--host', required=False, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api).')
@click.option('-i', '--port', required=False, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api).')
@click.option('--n_max', required=False, help='Number of items to list.')
@list_cmd.command('sessions')
@click.pass_context
def list_sessions(ctx, host, port, n_max):
    """
    return:
    - count: number of sessions
    - result: list of sessions
    """
    url = f'http://{host}:{port}/api/{API_VERSION}/sessions'
    headers = {}

    if n_max:
        headers['X-Limit'] = n_max

    click.echo(f'\nListing sessions: {url}\n')

    try:
        response = requests.get(url, headers=headers)
        print_response(response, 'sessions')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {host}:{port}')


@click.option('-h', '--host', required=False, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api).')
@click.option('-i', '--port', required=False, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api).')
@click.option('--n_max', required=False, help='Number of items to list.')
@list_cmd.command('statuses')
@click.pass_context
def list_statuses(ctx, host, port, n_max):
    """
    return:
    - count: number of statuses
    - result: list of statuses
    """
    url = f'http://{host}:{port}/api/{API_VERSION}/statuses'
    headers = {}

    if n_max:
        headers['X-Limit'] = n_max

    click.echo(f'\nListing statuses: {url}\n')

    try:
        response = requests.get(url, headers=headers)
        print_response(response, 'statuses')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {host}:{port}')


@click.option('-h', '--host', required=False, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api).')
@click.option('-i', '--port', required=False, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api).')
@click.option('--n_max', required=False, help='Number of items to list.')
@list_cmd.command('validations')
@click.pass_context
def list_validations(ctx, host, port, n_max):
    """
    return:
    - count: number of validations
    - result: list of validations
    """
    url = f'http://{host}:{port}/api/{API_VERSION}/validations'
    headers = {}

    if n_max:
        headers['X-Limit'] = n_max

    click.echo(f'\nListing validations: {url}\n')

    try:
        response = requests.get(url, headers=headers)
        print_response(response, 'validations')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {host}:{port}')