

import click
import requests

from .main import main
from .shared import API_VERSION, CONTROLLER_DEFAULTS, get_api_url, get_token


def print_response(response, entity_name: str):
    """
    Prints the api response to the cli.
    :param response:
        type: array
        description: list of entities
    :param entity_name:
        type: string
        description: name of entity
    return: None
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


@click.option('-p', '--protocol', required=False, default=CONTROLLER_DEFAULTS['protocol'], help='Communication protocol of controller (api)')
@click.option('-H', '--host', required=False, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api)')
@click.option('-P', '--port', required=False, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api)')
@click.option('-t', '--token', required=False, help='Authentication token')
@click.option('--n_max', required=False, help='Number of items to list')
@list_cmd.command('clients')
@click.pass_context
def list_clients(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None):
    """
    return:
    - count: number of clients
    - result: list of clients
    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint='clients')
    headers = {}

    if n_max:
        headers['X-Limit'] = n_max

    _token = get_token(token)

    if _token:
        headers['Authorization'] = _token

    click.echo(f'\nListing clients: {url}\n')
    click.echo(f'Headers: {headers}')

    try:
        response = requests.get(url, headers=headers)
        print_response(response, 'clients')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {url}')


@click.option('-p', '--protocol', required=False, default=CONTROLLER_DEFAULTS['protocol'], help='Communication protocol of controller (api)')
@click.option('-H', '--host', required=False, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api)')
@click.option('-P', '--port', required=False, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api)')
@click.option('-t', '--token', required=False, help='Authentication token')
@click.option('--n_max', required=False, help='Number of items to list')
@list_cmd.command('combiners')
@click.pass_context
def list_combiners(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None):
    """
    return:
    - count: number of combiners
    - result: list of combiners
    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint='combiners')
    headers = {}

    if n_max:
        headers['X-Limit'] = n_max

    _token = get_token(token)

    if _token:
        headers['Authorization'] = _token

    click.echo(f'\nListing combiners: {url}\n')
    click.echo(f'Headers: {headers}')

    try:
        response = requests.get(url, headers=headers)
        print_response(response, 'combiners')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {url}')


@click.option('-p', '--protocol', required=False, default=CONTROLLER_DEFAULTS['protocol'], help='Communication protocol of controller (api)')
@click.option('-H', '--host', required=False, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api)')
@click.option('-P', '--port', required=False, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api)')
@click.option('-t', '--token', required=False, help='Authentication token')
@click.option('--n_max', required=False, help='Number of items to list')
@list_cmd.command('models')
@click.pass_context
def list_models(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None):
    """
    return:
    - count: number of models
    - result: list of models
    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint='models')
    headers = {}

    if n_max:
        headers['X-Limit'] = n_max

    _token = get_token(token)

    if _token:
        headers['Authorization'] = _token

    click.echo(f'\nListing models: {url}\n')
    click.echo(f'Headers: {headers}')

    try:
        response = requests.get(url, headers=headers)
        print_response(response, 'models')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {url}')


@click.option('-p', '--protocol', required=False, default=CONTROLLER_DEFAULTS['protocol'], help='Communication protocol of controller (api)')
@click.option('-H', '--host', required=False, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api)')
@click.option('-P', '--port', required=False, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api)')
@click.option('-t', '--token', required=False, help='Authentication token')
@click.option('--n_max', required=False, help='Number of items to list')
@list_cmd.command('packages')
@click.pass_context
def list_packages(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None):
    """
    return:
    - count: number of packages
    - result: list of packages
    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint='packages')
    headers = {}

    if n_max:
        headers['X-Limit'] = n_max

    _token = get_token(token)

    if _token:
        headers['Authorization'] = _token

    click.echo(f'\nListing packages: {url}\n')
    click.echo(f'Headers: {headers}')

    try:
        response = requests.get(url, headers=headers)
        print_response(response, 'packages')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {url}')


@click.option('-p', '--protocol', required=False, default=CONTROLLER_DEFAULTS['protocol'], help='Communication protocol of controller (api)')
@click.option('-H', '--host', required=False, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api)')
@click.option('-P', '--port', required=False, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api)')
@click.option('-t', '--token', required=False, help='Authentication token')
@click.option('--n_max', required=False, help='Number of items to list')
@list_cmd.command('rounds')
@click.pass_context
def list_rounds(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None):
    """
    return:
    - count: number of rounds
    - result: list of rounds
    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint='rounds')
    headers = {}

    if n_max:
        headers['X-Limit'] = n_max

    _token = get_token(token)

    if _token:
        headers['Authorization'] = _token

    click.echo(f'\nListing rounds: {url}\n')
    click.echo(f'Headers: {headers}')

    try:
        response = requests.get(url, headers=headers)
        print_response(response, 'rounds')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {url}')


@click.option('-p', '--protocol', required=False, default=CONTROLLER_DEFAULTS['protocol'], help='Communication protocol of controller (api)')
@click.option('-H', '--host', required=False, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api)')
@click.option('-P', '--port', required=False, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api)')
@click.option('-t', '--token', required=False, help='Authentication token')
@click.option('--n_max', required=False, help='Number of items to list')
@list_cmd.command('sessions')
@click.pass_context
def list_sessions(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None):
    """
    return:
    - count: number of sessions
    - result: list of sessions
    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint='sessions')
    headers = {}

    if n_max:
        headers['X-Limit'] = n_max

    _token = get_token(token)

    if _token:
        headers['Authorization'] = _token

    click.echo(f'\nListing sessions: {url}\n')
    click.echo(f'Headers: {headers}')

    try:
        response = requests.get(url, headers=headers)
        print_response(response, 'sessions')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {url}')


@click.option('-p', '--protocol', required=False, default=CONTROLLER_DEFAULTS['protocol'], help='Communication protocol of controller (api)')
@click.option('-H', '--host', required=False, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api)')
@click.option('-P', '--port', required=False, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api)')
@click.option('-t', '--token', required=False, help='Authentication token')
@click.option('--n_max', required=False, help='Number of items to list')
@list_cmd.command('statuses')
@click.pass_context
def list_statuses(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None):
    """
    return:
    - count: number of statuses
    - result: list of statuses
    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint='statuses')
    headers = {}

    if n_max:
        headers['X-Limit'] = n_max

    _token = get_token(token)

    if _token:
        headers['Authorization'] = _token

    click.echo(f'\nListing statuses: {url}\n')
    click.echo(f'Headers: {headers}')

    try:
        response = requests.get(url, headers=headers)
        print_response(response, 'statuses')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {url}')


@click.option('-p', '--protocol', required=False, default=CONTROLLER_DEFAULTS['protocol'], help='Communication protocol of controller (api)')
@click.option('-H', '--host', required=False, default=CONTROLLER_DEFAULTS['host'], help='Hostname of controller (api)')
@click.option('-P', '--port', required=False, default=CONTROLLER_DEFAULTS['port'], help='Port of controller (api)')
@click.option('-t', '--token', required=False, help='Authentication token')
@click.option('--n_max', required=False, help='Number of items to list')
@list_cmd.command('validations')
@click.pass_context
def list_validations(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None):
    """
    return:
    - count: number of validations
    - result: list of validations
    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint='validations')
    headers = {}

    if n_max:
        headers['X-Limit'] = n_max

    _token = get_token(token)

    if _token:
        headers['Authorization'] = _token

    click.echo(f'\nListing validations: {url}\n')
    click.echo(f'Headers: {headers}')

    try:
        response = requests.get(url, headers=headers)
        print_response(response, 'validations')
    except requests.exceptions.ConnectionError:
        click.echo(f'Error: Could not connect to {url}')
