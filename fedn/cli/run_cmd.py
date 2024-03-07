import uuid

import click
import yaml

from fedn.common.exceptions import InvalidClientConfig
from fedn.common.log_config import logger
from fedn.network.clients.client import Client
from fedn.network.combiner.combiner import Combiner

from .main import main


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


@main.group('run')
@click.pass_context
def run_cmd(ctx):
    """

    :param ctx:
    """
    pass


@run_cmd.command('client')
@click.option('-d', '--discoverhost', required=False, help='Hostname for discovery services(reducer).')
@click.option('-p', '--discoverport', required=False, help='Port for discovery services (reducer).')
@click.option('--token', required=False, help='Set token provided by reducer if enabled')
@click.option('-n', '--name', required=False, default="client" + str(uuid.uuid4())[:8])
@click.option('-i', '--client_id', required=False)
@click.option('--local-package', is_flag=True, help='Enable local compute package')
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
def client_cmd(ctx, discoverhost, discoverport, token, name, client_id, local_package, force_ssl, dry_run, secure, preshared_cert,
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
    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'name': name,
              'client_id': client_id, 'remote_compute_context': remote, 'force_ssl': force_ssl, 'dry_run': dry_run, 'secure': secure,
              'preshared_cert': preshared_cert, 'verify': verify, 'preferred_combiner': preferred_combiner,
              'validator': validator, 'trainer': trainer, 'init': init, 'logfile': logfile, 'heartbeat_interval': heartbeat_interval,
              'reconnect_after_missed_heartbeat': reconnect_after_missed_heartbeat, 'verbosity': verbosity}

    if init:
        apply_config(config)

    validate_client_config(config)

    client = Client(config)
    client.run()


@run_cmd.command('combiner')
@click.option('-d', '--discoverhost', required=False, help='Hostname for discovery services (reducer).')
@click.option('-p', '--discoverport', required=False, help='Port for discovery services (reducer).')
@click.option('-t', '--token', required=False, help='Set token provided by reducer if enabled')
@click.option('-n', '--name', required=False, default="combiner" + str(uuid.uuid4())[:8], help='Set name for combiner.')
@click.option('-h', '--host', required=False, default="combiner", help='Set hostname.')
@click.option('-i', '--port', required=False, default=12080, help='Set port.')
@click.option('-f', '--fqdn', required=False, default=None, help='Set fully qualified domain name')
@click.option('-s', '--secure', is_flag=True, help='Enable SSL/TLS encrypted gRPC channels.')
@click.option('-v', '--verify', is_flag=True, help='Verify SSL/TLS for REST discovery service (reducer)')
@click.option('-c', '--max_clients', required=False, default=30, help='The maximal number of client connections allowed.')
@click.option('-in', '--init', required=False, default=None,
              help='Path to configuration file to (re)init combiner.')
@click.pass_context
def combiner_cmd(ctx, discoverhost, discoverport, token, name, host, port, fqdn, secure, verify, max_clients, init):
    """

    :param ctx:
    :param discoverhost:
    :param discoverport:
    :param token:
    :param name:
    :param hostname:
    :param port:
    :param secure:
    :param max_clients:
    :param init:
    """
    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'host': host,
              'port': port, 'fqdn': fqdn, 'name': name, 'secure': secure, 'verify': verify, 'max_clients': max_clients,
              'init': init}

    if config['init']:
        apply_config(config)

    combiner = Combiner(config)
    combiner.run()
