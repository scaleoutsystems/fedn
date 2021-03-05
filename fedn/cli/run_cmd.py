import click
import uuid
import yaml
from deprecated import deprecated

from .main import main

@deprecated(version='0.2.0', reason="You should switch to using a configuration file.")
def get_statestore_config_from_env():
    import os
    config = {
        'type': 'MongoDB',
        'mongo_config': {
            'username': os.environ.get('FEDN_MONGO_USER', 'default'),
            'password': os.environ.get('FEDN_MONGO_PASSWORD', 'password'),
            'host': os.environ.get('FEDN_MONGO_HOST', 'localhost'),
            'port': int(os.environ.get('FEDN_MONGO_PORT', '27017')),
        },
    }
    return config

@deprecated(version='0.2.0', reason="You should switch to using a configuration file.")
def get_storage_config_from_env():
    """ Deprecated. """
    import os
    s3_config = {
        'storage_type': 'S3',
        'storage_config': {
            'storage_access_key': os.environ['FEDN_MINIO_ACCESS_KEY'],
            'storage_secret_key': os.environ['FEDN_MINIO_SECRET_KEY'],
            'storage_bucket': 'fednmodels',
            'storage_secure_mode': False,
            'storage_hostname': os.environ['FEDN_MINIO_HOST'],
            'storage_port': int(os.environ['FEDN_MINIO_PORT'])
            },
        }
    return s3_config

@deprecated(version='0.2.0', reason="You should switch to using a configuration file.")
def get_network_id_from_env():
    """ Deprecated. """
    import os
    network_id = os.environ['ALLIANCE_UID']
    return network_id


def get_statestore_config_from_file(init):
    with open(init, 'r') as file:
        try:
            settings = dict(yaml.safe_load(file))
            return settings
        except yaml.YAMLError as e:
            raise(e)


@main.group('run')
@click.pass_context
def run_cmd(ctx):
    # if daemon:
    #    print('{} NYI should run as daemon...'.format(__file__))
    pass


@run_cmd.command('client')
@click.option('-d', '--discoverhost', required=False)
@click.option('-p', '--discoverport', required=False)
@click.option('-t', '--token', required=False)
@click.option('-n', '--name', required=False, default=str(uuid.uuid4()))
@click.option('-i', '--client_id', required=False)
@click.option('-r', '--remote', required=False, default=True, help='Enable remote configured execution context')
@click.option('-u', '--dry-run', required=False, default=False)
@click.option('-s', '--secure', required=False, default=True)
@click.option('-v', '--preshared-cert', required=False, default=False)
@click.option('-v', '--verify-cert', required=False, default=False)
@click.option('-c', '--preferred-combiner', required=False, default=False)
@click.option('-in', '--init', required=False, default=None, help='Set to a filename to (re)init client from file state.')
@click.pass_context
def client_cmd(ctx, discoverhost, discoverport, token, name, client_id, remote, dry_run, secure, preshared_cert,
               verify_cert,preferred_combiner, init):

    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'name': name,
              'client_id': client_id, 'remote_compute_context': remote, 'dry_run': dry_run, 'secure': secure,
              'preshared_cert': preshared_cert, 'verify_cert': verify_cert,'preferred_combiner':preferred_combiner, 'init':init}

    if config['init']:
        with open(config['init'], 'r') as file:
            try:
                settings = dict(yaml.safe_load(file))
            except yaml.YAMLError as e:
                print('Failed to read config from settings file, exiting.',flush=True)
                raise(e)

        # Read/overide settings from config file
        if 'controller' in settings:
            reducer_config = settings['controller']
            for key,val in reducer_config.items():
                config[key] = val

    from fedn.client import Client
    client = Client(config)
    client.run()


@run_cmd.command('reducer')
@click.option('-d', '--discoverhost', required=False)
@click.option('-p', '--discoverport', required=False)
@click.option('-t', '--token', required=False,default="reducer_token")
@click.option('-n', '--name', required=False, default=str(uuid.uuid4()))
@click.option('-i', '--init', required=True, default=None, help='Set to a filename to (re)init reducer from file state.')
@click.pass_context
def reducer_cmd(ctx, discoverhost, discoverport, token, name, init):
    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'name': name, 'init': init}

    # Read settings from config file
    try:
        fedn_config = get_statestore_config_from_file(config['init'])
    # Todo: Make more specific
    except Exception as e:
        print('Failed to read config from settings file, exiting.',flush=True)
        print(e,flush=True)
        exit(-1)

    try:
        network_id = fedn_config['network_id']
    except KeyError:
        print("No network_id in config, please specify the control network id.",flush=True)
        exit(-1)

    statestore_config = fedn_config['statestore']
    if statestore_config['type'] == 'MongoDB':
        from fedn.clients.reducer.statestore.mongoreducerstatestore import MongoReducerStateStore
        statestore = MongoReducerStateStore(network_id, statestore_config['mongo_config'], defaults=config['init'])
    else:
        print("Unsupported statestore type, exiting. ",flush=True)
        exit(-1)

    try:
        statestore.set_reducer(config)
    except:
        print("Failed to set reducer config in statestore, exiting.",flush=True)
        exit(-1)

    try:
        statestore.set_storage_backend(fedn_config['storage'])
    except KeyError:
        print("storage configuration missing in statestore_config.",flush=True)
        exit(-1)
    except:
        print("Failed to set storage config in statestore, exiting.",flush=True)
        exit(-1)

    # Control config
    control_config = fedn_config['control']
    print("CONTROL_CONFIG: ",control_config,flush=True)
    try:
        statestore.set_round_config(control_config)
    except:
        print("Failed to set control config, exiting.",flush=True)
        exit(-1)


    from fedn.reducer import Reducer
    reducer = Reducer(statestore)
    reducer.run()

@run_cmd.command('combiner')
@click.option('-d', '--discoverhost', required=False)
@click.option('-p', '--discoverport', required=False)
@click.option('-t', '--token', required=False)
@click.option('-n', '--name', required=False, default=str(uuid.uuid4()))
@click.option('-h', '--hostname', required=False,default="combiner")
@click.option('-i', '--port', required=False,default=12080)
@click.option('-s', '--secure', required=False, default=True)
@click.option('-c', '--max_clients', required=False, default=30)
@click.option('-in', '--init', required=False, default=None, help='Set to a filename to (re)init combiner from file state.')
@click.pass_context
def combiner_cmd(ctx, discoverhost, discoverport, token, name, hostname, port, secure, max_clients,init):
    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'myhost': hostname,
              'myport': port, 'myname': name, 'secure': secure, 'max_clients': max_clients,'init':init}


    if config['init']:
        with open(config['init'], 'r') as file:
            try:
                settings = dict(yaml.safe_load(file))
            except yaml.YAMLError as e:
                print('Failed to read config from settings file, exiting.',flush=True)
                raise(e)
        # Read/overide settings from config file
        if 'controller' in settings:
            controller_config = settings['controller']
            for key,val in controller_config.items():
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

@run_cmd.command('monitor')
@click.option('-h', '--combinerhost', required=False)
@click.option('-p', '--combinerport', required=False)
@click.option('-t', '--token', required=True)
@click.option('-n', '--name', required=False, default="monitor")
@click.option('-s', '--secure', required=False, default=False)
def monitor_cmd(ctx, combinerhost, combinerport, token, name, secure):
    import os
    if not combinerhost:
        combinerhost = os.environ['MONITOR_HOST']
    if not combinerport:
        combinerport = os.environ['MONITOR_PORT']

    config = {'host': combinerhost, 'port': combinerport, 'token': token, 'name': name, 'secure': secure}
    from fedn.monitor import Monitor

    monitor = Monitor(config)
    monitor.run()
