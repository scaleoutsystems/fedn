import click
import uuid
import yaml
from deprecated import deprecated

from .main import main

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

    from fednclient.client import Client
    client = Client(config)
    client.run()
