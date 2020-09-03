import click

from .main import main

@main.group('run')
@click.pass_context
def run_cmd(ctx):
    #if daemon:
    #    print('{} NYI should run as daemon...'.format(__file__))
    pass


@run_cmd.command('client')
@click.option('-d', '--discoverhost', required=True)
@click.option('-p', '--discoverport', required=True)
@click.option('-t', '--token', required=True)
@click.option('-n', '--name', required=False, default=None)
@click.option('-i', '--client_id', required=True,default=None)
@click.pass_context
def client_cmd(ctx, discoverhost, discoverport, token, name, client_id):

    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'name': name, 'client_id': client_id }

    from fedn.client import Client
    client = Client(config)
    client.run()

@run_cmd.command('reducer')
@click.option('-d', '--discoverhost', required=False)
@click.option('-p', '--discoverport', required=False)
@click.option('-t', '--token', required=True)
@click.option('-n', '--name', required=False, default=None)
@click.pass_context
def reducer_cmd(ctx, discoverhost, discoverport, token, name):

    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'name': name}

    from fedn.reducer import Reducer
    reducer = Reducer(config)
    reducer.run()

@run_cmd.command('combiner')
@click.option('-d', '--discoverhost', required=True)
@click.option('-p', '--discoverport', required=True)
@click.option('-t', '--token', required=True)
@click.option('-n', '--name', required=False, default=None)
@click.option('-h', '--hostname', required=True)
@click.option('-i', '--port', required=True)
@click.pass_context
def combiner_cmd(ctx, discoverhost, discoverport, token, name, hostname, port):

    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'myhost': hostname,
              'myport': port, 'myname': name}

    from fedn.combiner import Combiner
    combiner = Combiner(config)
    combiner.run()
