import time
import uuid

import click
import yaml

from fedn.client import Client
from fedn.clients.reducer.restservice import (decode_auth_token,
                                              encode_auth_token)
from fedn.clients.reducer.statestore.mongoreducerstatestore import \
    MongoReducerStateStore
from fedn.combiner import Combiner
from fedn.reducer import Reducer

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
        print("--local-package was used, but no helper was found in --init settings file.", flush=True)
        exit(-1)
    return helper


@main.group('run')
@click.pass_context
def run_cmd(ctx):
    """

    :param ctx:
    """
    # if daemon:
    #    print('{} NYI should run as daemon...'.format(__file__))
    pass


@run_cmd.command('client')
@click.option('-d', '--discoverhost', required=False)
@click.option('-p', '--discoverport', required=False)
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
@click.option('-l', '--logfile', required=False, default='{}-client.log'.format(time.strftime("%Y%m%d-%H%M%S")),
              help='Set logfile for client log to file.')
@click.option('--heartbeat-interval', required=False, default=2)
@click.option('--reconnect-after-missed-heartbeat', required=False, default=30)
@click.pass_context
def client_cmd(ctx, discoverhost, discoverport, token, name, client_id, local_package, force_ssl, dry_run, secure, preshared_cert,
               verify, preferred_combiner, validator, trainer, init, logfile, heartbeat_interval, reconnect_after_missed_heartbeat):
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
    :return:
    """
    remote = False if local_package else True
    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'name': name,
              'client_id': client_id, 'remote_compute_context': remote, 'force_ssl': force_ssl, 'dry_run': dry_run, 'secure': secure,
              'preshared_cert': preshared_cert, 'verify': verify, 'preferred_combiner': preferred_combiner,
              'validator': validator, 'trainer': trainer, 'init': init, 'logfile': logfile, 'heartbeat_interval': heartbeat_interval,
              'reconnect_after_missed_heartbeat': reconnect_after_missed_heartbeat}

    if config['init']:
        with open(config['init'], 'r') as file:
            try:
                settings = dict(yaml.safe_load(file))
            except Exception:
                print('Failed to read config from settings file, exiting.', flush=True)
                return
                # raise(e)

        # Read/overide settings from config file
        if 'controller' in settings:
            reducer_config = settings['controller']
            for key, val in reducer_config.items():
                config[key] = val

    try:
        if config['discover_host'] is None or \
                config['discover_host'] == '':
            print(
                "Missing required configuration: discover_host", flush=True)
            return
        if 'discover_port' not in config.keys():
            config['discover_port'] = None
    except Exception:
        print("Could not load config appropriately. Check config", flush=True)
        return

    client = Client(config)
    client.run()


@run_cmd.command('reducer')
@click.option('-h', '--host', required=False)
@click.option('-p', '--port', required=False, default='8090', show_default=True)
@click.option('-k', '--secret-key', required=False, help='Set secret key to enable jwt token authentication.')
@click.option('-l', '--local-package', is_flag=True, help='Enable use of local compute package')
@click.option('-n', '--name', required=False, default="reducer" + str(uuid.uuid4())[:8], help='Set service name')
@click.option('-i', '--init', required=True, default=None,
              help='Set to a filename to (re)init reducer state from file.')
@click.pass_context
def reducer_cmd(ctx, host, port, secret_key, local_package, name, init):
    """

    :param ctx:
    :param discoverhost:
    :param discoverport:
    :param token:
    :param name:
    :param init:
    """
    remote = False if local_package else True
    config = {'host': host, 'port': port, 'secret_key': secret_key,
              'name': name, 'remote_compute_context': remote, 'init': init}

    # Read settings from config file
    try:
        fedn_config = get_statestore_config_from_file(config['init'])
    except Exception as e:
        print('Failed to read config from settings file, exiting.', flush=True)
        print(e, flush=True)
        exit(-1)

    if not remote:
        _ = check_helper_config_file(fedn_config)

    try:
        network_id = fedn_config['network_id']
    except KeyError:
        print("No network_id in config, please specify the control network id.", flush=True)
        exit(-1)

    # Obtain state from database, in case already initialized (service restart)
    statestore_config = fedn_config['statestore']
    if statestore_config['type'] == 'MongoDB':
        statestore = MongoReducerStateStore(
            network_id, statestore_config['mongo_config'], defaults=config['init'])
    else:
        print("Unsupported statestore type, exiting. ", flush=True)
        exit(-1)

    # Enable JWT token authentication.
    if config['secret_key']:
        # If we already have a valid token in statestore config, use that one.
        existing_config = statestore.get_reducer()
        if existing_config:
            try:
                existing_config = statestore.get_reducer()
                current_token = existing_config['token']
                status = decode_auth_token(current_token, config['secret_key'])
                if status != 'Success':
                    token = encode_auth_token(config['secret_key'])
                    config['token'] = token
            except Exception:
                raise

        else:
            token = encode_auth_token(config['secret_key'])
            config['token'] = token
    try:
        statestore.set_reducer(config)
    except Exception:
        print("Failed to set reducer config in statestore, exiting.", flush=True)
        exit(-1)

    # Configure storage backend (currently supports MinIO)
    try:
        statestore.set_storage_backend(fedn_config['storage'])
    except KeyError:
        print("storage configuration missing in statestore_config.", flush=True)
        exit(-1)
    except Exception:
        print("Failed to set storage config in statestore, exiting.", flush=True)
        exit(-1)

    # Configure controller
    control_config = fedn_config['control']
    try:
        statestore.set_round_config(control_config)
    except Exception:
        print("Failed to set control config, exiting.", flush=True)
        exit(-1)

    reducer = Reducer(statestore)
    reducer.run()


@run_cmd.command('combiner')
@click.option('-d', '--discoverhost', required=False, help='Hostname for discovery services (reducer).')
@click.option('-p', '--discoverport', required=False, help='Port for discovery services (reducer).')
@click.option('-t', '--token', required=False, help='Specify token for connecting to the reducer.')
@click.option('-n', '--name', required=False, default="combiner" + str(uuid.uuid4())[:8], help='Set name for combiner.')
@click.option('-h', '--host', required=False, default="combiner", help='Set hostname.')
@click.option('-i', '--port', required=False, default=12080, help='Set port.')
@click.option('-s', '--secure', is_flag=True, help='Enable SSL/TLS encrypted gRPC channels.')
@click.option('-v', '--verify', is_flag=True, help='Verify SSL/TLS for REST service')
@click.option('-c', '--max_clients', required=False, default=30, help='The maximal number of client connections allowed.')
@click.option('-in', '--init', required=False, default=None,
              help='Path to configuration file to (re)init combiner.')
@click.pass_context
def combiner_cmd(ctx, discoverhost, discoverport, token, name, host, port, secure, verify, max_clients, init):
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
    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'myhost': host,
              'myport': port, 'myname': name, 'secure': secure, 'verify': verify, 'max_clients': max_clients, 'init': init}

    if config['init']:
        with open(config['init'], 'r') as file:
            try:
                settings = dict(yaml.safe_load(file))
            except yaml.YAMLError as e:
                print('Failed to read config from settings file, exiting.', flush=True)
                raise (e)

        # Read/overide settings from config file
        if 'controller' in settings:
            controller_config = settings['controller']
            for key, val in controller_config.items():
                config[key] = val

        if 'combiner' in settings:
            combiner_config = settings['combiner']
            config['myname'] = combiner_config['name']
            config['myhost'] = combiner_config['host']
            if 'fqdn' in combiner_config.keys():
                config['fqdn'] = combiner_config['fqdn']
            else:
                config['fqdn'] = None
            config['myport'] = combiner_config['port']
            config['max_clients'] = combiner_config['max_clients']

    combiner = Combiner(config)
    combiner.run()
