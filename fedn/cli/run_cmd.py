import click
import uuid
import yaml
import time
from fedn.clients.reducer.restservice import encode_auth_token, decode_auth_token
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
@click.option('-u', '--dry-run', required=False, default=False)
@click.option('-s', '--secure', required=False, default=True)
@click.option('-pc', '--preshared-cert', required=False, default=False)
@click.option('-v', '--verify-cert', required=False, default=False)
@click.option('-c', '--preferred-combiner', required=False, default=False)
@click.option('-va', '--validator', required=False, default=True)
@click.option('-tr', '--trainer', required=False, default=True)
@click.option('-in', '--init', required=False, default=None,
              help='Set to a filename to (re)init client from file state.')
@click.option('-l', '--logfile', required=False, default='{}-client.log'.format(time.strftime("%Y%m%d-%H%M%S")),
              help='Set logfile for client log to file.')
@click.option('--heartbeat-interval',required=False, default=2)
@click.option('--reconnect-after-missed-heartbeat',required=False, default=30)
@click.pass_context
def client_cmd(ctx, discoverhost, discoverport, token, name, client_id, local_package, dry_run, secure, preshared_cert,
               verify_cert, preferred_combiner, validator, trainer, init, logfile, heartbeat_interval, reconnect_after_missed_heartbeat):
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
              'client_id': client_id, 'remote_compute_context': remote, 'dry_run': dry_run, 'secure': secure,
              'preshared_cert': preshared_cert, 'verify_cert': verify_cert, 'preferred_combiner': preferred_combiner,
              'validator': validator, 'trainer': trainer, 'init': init, 'logfile': logfile,'heartbeat_interval': heartbeat_interval,
              'reconnect_after_missed_heartbeat': 30}

    if config['init']:
        with open(config['init'], 'r') as file:
            try:
                settings = dict(yaml.safe_load(file))
            except Exception as e:
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
                config['discover_host'] == '' or \
                config['discover_host'] is None or \
                config['discover_port'] == '':
            print("Missing required configuration: discover_host, discover_port", flush=True)
            return
    except Exception as e:
        print("Could not load config appropriately. Check config", flush=True)
        return

    from fedn.client import Client
    client = Client(config)
    client.run()


@run_cmd.command('reducer')
@click.option('-d', '--discoverhost', required=False)
@click.option('-p', '--discoverport', required=False, default='8090', show_default=True)
@click.option('-s', '--secret-key', required=False, help='Set secret key to enable jwt token authentication.')
@click.option('-l', '--local-package', is_flag=True, help='Enable local compute package')
@click.option('-n', '--name', required=False, default="reducer" + str(uuid.uuid4())[:8])
@click.option('-i', '--init', required=True, default=None,
              help='Set to a filename to (re)init reducer from file state.')
@click.pass_context
def reducer_cmd(ctx, discoverhost, discoverport, secret_key, local_package, name, init):
    """

    :param ctx:
    :param discoverhost:
    :param discoverport:
    :param token:
    :param name:
    :param init:
    """
    remote = False if local_package else True
    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'secret_key': secret_key, 'name': name, 
              'remote_compute_context': remote, 'init': init}

    # Read settings from config file
    try:
        fedn_config = get_statestore_config_from_file(config['init'])
    # Todo: Make more specific
    except Exception as e:
        print('Failed to read config from settings file, exiting.', flush=True)
        print(e, flush=True)
        exit(-1)
    
    if not remote:
        helper = check_helper_config_file(fedn_config)
        
    try:
        network_id = fedn_config['network_id']
    except KeyError:
        print("No network_id in config, please specify the control network id.", flush=True)
        exit(-1)

    statestore_config = fedn_config['statestore']
    if statestore_config['type'] == 'MongoDB':
        from fedn.clients.reducer.statestore.mongoreducerstatestore import MongoReducerStateStore
        statestore = MongoReducerStateStore(network_id, statestore_config['mongo_config'], defaults=config['init'])
    else:
        print("Unsupported statestore type, exiting. ", flush=True)
        exit(-1)


    if config['secret_key']:
        # If we already have a valid token in statestore config, use that one. 
        existing_config = statestore.get_reducer()
        if existing_config: 
            try:
                existing_config = statestore.get_reducer()
                current_token = existing_config['token']
                status = decode_auth_token(current_token,config['secret_key'])
                if status != 'Success':
                    token = encode_auth_token(config['secret_key'])
                    config['token'] = token
            except:
                raise

        else:
            token = encode_auth_token(config['secret_key'])
            config['token'] = token

    try:
        statestore.set_reducer(config)
    except:
        print("Failed to set reducer config in statestore, exiting.", flush=True)
        exit(-1)

    try:
        statestore.set_storage_backend(fedn_config['storage'])
    except KeyError:
        print("storage configuration missing in statestore_config.", flush=True)
        exit(-1)
    except:
        print("Failed to set storage config in statestore, exiting.", flush=True)
        exit(-1)

    # Control config
    control_config = fedn_config['control']
    try:
        statestore.set_round_config(control_config)
    except:
        print("Failed to set control config, exiting.", flush=True)
        exit(-1)

    from fedn.reducer import Reducer
    reducer = Reducer(statestore)
    reducer.run()


@run_cmd.command('combiner')
@click.option('-d', '--discoverhost', required=False)
@click.option('-p', '--discoverport', required=False)
@click.option('-t', '--token', required=False)
@click.option('-n', '--name', required=False, default="combiner" + str(uuid.uuid4())[:8])
@click.option('-h', '--hostname', required=False, default="combiner")
@click.option('-i', '--port', required=False, default=12080)
@click.option('-s', '--secure', required=False, default=True)
@click.option('-c', '--max_clients', required=False, default=30)
@click.option('-in', '--init', required=False, default=None,
              help='Set to a filename to (re)init combiner from file state.')
@click.pass_context
def combiner_cmd(ctx, discoverhost, discoverport, token, name, hostname, port, secure, max_clients, init):
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
    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'myhost': hostname,
              'myport': port, 'myname': name, 'secure': secure, 'max_clients': max_clients, 'init': init}

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
            config['myport'] = combiner_config['port']
            config['max_clients'] = combiner_config['max_clients']

    from fedn.combiner import Combiner
    combiner = Combiner(config)
    combiner.run()
