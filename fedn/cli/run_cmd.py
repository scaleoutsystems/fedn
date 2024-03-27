import threading
import uuid

import click
import yaml

from fedn.common.exceptions import InvalidClientConfig
from fedn.common.log_config import logger
from fedn.network.clients.client import Client
from fedn.network.combiner.combiner import Combiner

from .main import main
from .shared import CLIENT_DEFAULTS, COMBINER_DEFAULTS, CONTROLLER_DEFAULTS, get_client_package_path


def get_statestore_config_from_file(init):
    """

    :param init:
    :return:
    """
    with open(init, 'r') as file:
        try:
            settings = dict(yaml.safe_load(file))
            return settings
        except yaml.YAMLError as e:
            raise (e)


def check_helper_config_file(config):
    control = config['control']
    try:
        helper = control["helper"]
    except KeyError:
        logger.error("--local-package was used, but no helper was found in --init settings file.")
        exit(-1)
    return helper


def apply_config(config):
    """Parse client config from file.

    Override configs from the CLI with settings in config file.

    :param config: Client config (dict).
    """
    with open(config['init'], 'r') as file:
        try:
            settings = dict(yaml.safe_load(file))
        except Exception:
            logger.error('Failed to read config from settings file, exiting.')
            return

    for key, val in settings.items():
        config[key] = val


def validate_client_config(config):
    """Validate client configuration.

    :param config: Client config (dict).
    """

    try:
        if config['discover_host'] is None or \
                config['discover_host'] == '':
            raise InvalidClientConfig("Missing required configuration: discover_host")
        if 'discover_port' not in config.keys():
            config['discover_port'] = None
    except Exception:
        raise InvalidClientConfig("Could not load config from file. Check config")


@main.group('run', invoke_without_command=True)
@click.pass_context
def run_cmd(ctx):
    """
    - Run FEDn apps (client, combiner, controller).
    """
    if ctx.invoked_subcommand is None:
        click.echo('Running FEDn...')
        config: dict = {'host': CONTROLLER_DEFAULTS['host'], 'port': CONTROLLER_DEFAULTS['port'], 'debug': CONTROLLER_DEFAULTS['debug']}
        click.echo(f"Starting API server on {config['host']}:{config['port']}")

        from fedn.network.api.server import start_api

        thread_api = threading.Thread(target=start_api, args=(config,))
        thread_api.start()

        config = {
            'discover_host': COMBINER_DEFAULTS['discover_host'],
            'discover_port': COMBINER_DEFAULTS['discover_port'],
            'token': None,
            'host': COMBINER_DEFAULTS['host'],
            'port': COMBINER_DEFAULTS['port'],
            'fqdn': None,
            'name': COMBINER_DEFAULTS['name'],
            'secure': False,
            'verify': False,
            'max_clients': COMBINER_DEFAULTS['max_clients'],
            'init': None
        }

        click.echo(f"Starting combiner on {config['host']}:{config['port']}")

        combiner = Combiner(config)
        thread_combiner = threading.Thread(target=combiner.run)
        thread_combiner.start()

        thread_api.join()
        thread_combiner.join()


@run_cmd.command('client')
@click.option('-d', '--discoverhost', default=CLIENT_DEFAULTS['discover_host'], required=False, help='Hostname for discovery services(reducer).')
@click.option('-p', '--discoverport', default=CLIENT_DEFAULTS['discover_port'], required=False, help='Port for discovery services (reducer).')
@click.option('--token', required=False, help='Set token provided by reducer if enabled')
@click.option('-n', '--name', required=False, default="client" + str(uuid.uuid4())[:8])
@click.option('-i', '--client_id', required=False)
@click.option('--local-package', is_flag=True, help='Enable local compute package')
@click.option('-pp', '--package-path', required=False, help='Path to local package')
@click.option('--force-ssl', is_flag=True, help='Force SSL/TLS for REST service')
@click.option('-u', '--dry-run', required=False, default=False)
@click.option('-s', '--secure', required=False, default=False)
@click.option('-pc', '--preshared-cert', required=False, default=False)
@click.option('-v', '--verify', is_flag=True, help='Verify SSL/TLS for REST service')
@click.option('-c', '--preferred-combiner', required=False, default=False)
@click.option('-va', '--validator', required=False, default=True)
@click.option('-tr', '--trainer', required=False, default=True)
@click.option('-in', '--init', required=False, default=None,
              help='Set to a filename to (re)init client from file state.')
@click.option('-l', '--logfile', required=False, default=None,
              help='Set logfile for client log to file.')
@click.option('--heartbeat-interval', required=False, default=2)
@click.option('--reconnect-after-missed-heartbeat', required=False, default=30)
@click.option('--verbosity', required=False, default='INFO', type=click.Choice(['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'], case_sensitive=False))
@click.pass_context
def client_cmd(ctx, discoverhost, discoverport, token, name, client_id, local_package, package_path, force_ssl, dry_run, secure, preshared_cert,
               verify, preferred_combiner, validator, trainer, init, logfile, heartbeat_interval, reconnect_after_missed_heartbeat,
               verbosity):
    """

    :param ctx:
    :param discoverhost:
    :param discoverport:
    :param token:
    :param name:
    :param client_id:
    :param remote:
    :param dry_run:
    :param secure:
    :param preshared_cert:
    :param verify_cert:
    :param preferred_combiner:
    :param init:
    :param logfile:
    :param hearbeat_interval
    :param reconnect_after_missed_heartbeat
    :param verbosity
    :return:
    """
    remote = False if local_package else True

    package_path = get_client_package_path(package_path)

    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'name': name,
              'client_id': client_id, 'remote_compute_context': remote, 'force_ssl': force_ssl, 'dry_run': dry_run, 'secure': secure,
              'preshared_cert': preshared_cert, 'verify': verify, 'preferred_combiner': preferred_combiner,
              'validator': validator, 'trainer': trainer, 'init': init, 'logfile': logfile, 'heartbeat_interval': heartbeat_interval,
              'reconnect_after_missed_heartbeat': reconnect_after_missed_heartbeat, 'verbosity': verbosity, 'package_path': package_path}

    if init:
        apply_config(config)

    validate_client_config(config)

    client = Client(config)
    client.run()


@run_cmd.command('combiner')
@click.option('-d', '--discoverhost', required=False, default=COMBINER_DEFAULTS['discover_host'], help='Hostname for discovery services (reducer).')
@click.option('-p', '--discoverport', required=False, default=COMBINER_DEFAULTS['discover_port'], help='Port for discovery services (reducer).')
@click.option('-t', '--token', required=False, help='Set token provided by reducer if enabled')
@click.option('-n', '--name', required=False, default=COMBINER_DEFAULTS["name"] + str(uuid.uuid4())[:8], help='Set name for combiner.')
@click.option('-h', '--host', required=False, default=COMBINER_DEFAULTS['host'], help='Set hostname.')
@click.option('-i', '--port', required=False, default=COMBINER_DEFAULTS['port'], help='Set port.')
@click.option('-f', '--fqdn', required=False, default=None, help='Set fully qualified domain name')
@click.option('-s', '--secure', is_flag=True, help='Enable SSL/TLS encrypted gRPC channels.')
@click.option('-v', '--verify', is_flag=True, help='Verify SSL/TLS for REST discovery service (reducer)')
@click.option('-c', '--max_clients', required=False, default=COMBINER_DEFAULTS["max_clients"], help='The maximal number of client connections allowed.')
@click.option('-in', '--init', required=False, default=None,
              help='Path to configuration file to (re)init combiner.')
@click.pass_context
def combiner_cmd(ctx, discoverhost, discoverport, token, name, host, port, fqdn, secure, verify, max_clients, init):
    """

    :param ctx:
    :param discoverhost:
        type: str
    :param discoverport:
        type: int
    :param token:
        type: str
    :param name:
        type: str
    :param hostname:
        type: str
    :param port:
        type: int
    :param secure:
        type: bool
    :param max_clients:
        type: int
    :param init:
        type: str (path to file with configuration settings for combiner)
    """
    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'host': host,
              'port': port, 'fqdn': fqdn, 'name': name, 'secure': secure, 'verify': verify, 'max_clients': max_clients,
              'init': init}

    if config['init']:
        apply_config(config)

    click.echo(f"Starting combiner on {host}:{port}")

    combiner = Combiner(config)
    combiner.run()


@run_cmd.command('controller')
@click.option('-h', '--host', required=False, default=CONTROLLER_DEFAULTS['host'], help='Set hostname.')
@click.option('-i', '--port', required=False, default=CONTROLLER_DEFAULTS['port'], help='Set port.')
@click.option('--debug', required=False, default=CONTROLLER_DEFAULTS['debug'], help='Set debug.')
@click.option('-in', '--init', required=False, default=None,
              help='Path to configuration file to (re)init controller.')
@click.pass_context
def controller_cmd(ctx, host, port, debug, init):
    """

    :param ctx:
    :param host:
        type: str
    :param port:
        type: int
    :param debug:
        type: bool
    :param init:
        type: str (path to file with configuration settings for controller)
    """
    config = {'host': host, 'port': port, 'debug': debug, 'init': init}

    if config['init']:
        apply_config(config)

    click.echo(f"Starting API server on {host}:{port}")

    from fedn.network.api.server import start_api

    start_api(config)
