import click
import uuid

from .main import main


@main.group('run')
@click.pass_context
def run_cmd(ctx):
    # if daemon:
    #    print('{} NYI should run as daemon...'.format(__file__))
    pass


@run_cmd.command('client')
@click.option('-d', '--discoverhost', required=True)
@click.option('-p', '--discoverport', required=True)
@click.option('-t', '--token', required=True)
@click.option('-n', '--name', required=False, default=str(uuid.uuid4()))
@click.option('-i', '--client_id', required=False)
@click.option('-r', '--remote', required=False, default=True, help='Enable remote configured execution context')
@click.option('-u', '--dry-run', required=False, default=False)
@click.option('-s', '--secure', required=False, default=True)
@click.option('-v', '--preshared-cert', required=False, default=False)
@click.option('-v', '--verify-cert', required=False, default=False)
@click.pass_context
def client_cmd(ctx, discoverhost, discoverport, token, name, client_id, remote, dry_run, secure, preshared_cert,
               verify_cert):
    if name == None:
        import uuid
        name = str(uuid.uuid4())

    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'name': name,
              'client_id': client_id, 'remote_compute_context': remote, 'dry_run': dry_run, 'secure': secure,
              'preshared_cert': preshared_cert, 'verify_cert': verify_cert}

    from fedn.client import Client
    client = Client(config)
    client.run()


def get_statestore_config():
    import os
    config = {
        'type': 'MongoDB',
        'mongo_config': {
            'username': os.environ.get('FEDN_MONGO_USER', 'default'),
            'password': os.environ.get('FEDN_MONGO_PASSWORD', 'password'),
            'host': os.environ.get('FEDN_MONGO_HOST', 'localhost'),
            'port': int(os.environ.get('FEDN_MONGO_PORT', '27017')),
        }
    }
    return config

@run_cmd.command('reducer')
@click.option('-d', '--discoverhost', required=False)
@click.option('-p', '--discoverport', required=False)
@click.option('-t', '--token', required=True)
@click.option('-n', '--name', required=False, default=None)
@click.option('-i', '--init', required=False, default=None, help='Set to a filename to (re)init reducer from file state.')
@click.pass_context
def reducer_cmd(ctx, discoverhost, discoverport, token, name, init):
    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'name': name, 'init': init}

    # TODO: Move to init file, and add a separate CLI config command (fedn add storage)
    import os
    s3_config = {
        'storage_type': 'S3',
        'storage_access_key': os.environ['FEDN_MINIO_ACCESS_KEY'],
        'storage_secret_key': os.environ['FEDN_MINIO_SECRET_KEY'],
        'storage_bucket': 'fednmodels',
        'storage_secure_mode': False,
        'storage_hostname': os.environ['FEDN_MINIO_HOST'],
        'storage_port': int(os.environ['FEDN_MINIO_PORT'])
        }

    # TODO: Move to init file / additional configs
    statestore_config = get_statestore_config()

    # TODO: Move to cli argument and/or init file
    network_id = os.environ['ALLIANCE_UID']

    if statestore_config['type'] == 'MongoDB': 
        from fedn.clients.reducer.statestore.mongoreducerstatestore import MongoReducerStateStore
        statestore = MongoReducerStateStore(network_id, statestore_config['mongo_config'], defaults=config['init'])
    else:
        print("Unsupported statestore type, exiting. ",flush=True)
        raise
    
    try:
        statestore.set_reducer(config)
    except:
        print("Failed to set reducer config in statestore, exiting.",flush=True)
        raise
    
    try:
        statestore.set_storage_backend(s3_config)
    except:
        print("Failed to set storage config in statestore, exiting.",flush=True)
        raise

    from fedn.reducer import Reducer
    reducer = Reducer(statestore)
    reducer.run()


@run_cmd.command('combiner')
@click.option('-d', '--discoverhost', required=True)
@click.option('-p', '--discoverport', required=True)
@click.option('-t', '--token', required=True)
@click.option('-n', '--name', required=False, default=None)
@click.option('-h', '--hostname', required=True)
@click.option('-i', '--port', required=True)
@click.option('-s', '--secure', required=False, default=True)
@click.option('-c', '--max_clients', required=False, default=8)
@click.pass_context
def combiner_cmd(ctx, discoverhost, discoverport, token, name, hostname, port, secure, max_clients):
    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'myhost': hostname,
              'myport': port, 'myname': name, 'secure': secure, 'max_clients': max_clients}

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
