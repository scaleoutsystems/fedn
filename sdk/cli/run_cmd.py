import click

from .main import main


@click.option('--daemon',
              is_flag=True,
              help=(
                      "Specify to run in daemon mode."
              )
              )
@main.group('run')
@click.pass_context
def run_cmd(ctx, daemon):
    if daemon:
        print('{} NYI should run as daemon...'.format(__file__))


@run_cmd.command('client')
@click.option('-d', '--discoverhost', required=True)
@click.option('-p', '--discoverport', required=True)
@click.option('-t', '--token', required=True)
@click.option('-n', '--name', required=False, default=None)
@click.pass_context
def client_cmd(ctx, discoverhost, discoverport, token, name):

    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'name': name}

    project = ctx.obj['PROJECT']
    from fedn.member.client import Client
    client = Client(config)
    client.run()


@run_cmd.command('fedavg')
@click.pass_context
@click.option('-d', '--discoverhost', required=True)
@click.option('-p', '--discoverport', required=True)
@click.option('-h', '--hostname', required=True)
@click.option('-i', '--port', required=True)
@click.option('-t', '--token', required=True)
@click.option('-n', '--name', required=True)
def fedavg_cmd(ctx, discoverhost, discoverport, hostname, port, name, token):
    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'myhost': hostname,
              'myport': port, 'myname': name}

    project = ctx.obj['PROJECT']

    from fedn.combiner.helpers import get_combiner
    from fedn.combiner.server import FednServer
    server = FednServer(config, get_combiner)

    server.run(config)


@run_cmd.command('reducer')
@click.pass_context
def reducer_cmd(ctx, ):
    from fedn.reducer.reducer import Reducer
    project = ctx.obj['PROJECT']
    reducer = Reducer(project)

    reducer.run()
