import click
from .main import main

import requests


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
@click.option('-c', '--config', required=False, default='project.yaml')
@click.option('-n', '--name', required=False, default=None)
@click.pass_context
def client_cmd(ctx, config, name):
    project = ctx.obj['PROJECT']
    from fedn.member.client import Client
    client = Client(project)
    client.run()


@run_cmd.command('fedavg')
@click.pass_context
@click.option('-c', '--config', required=False, default='project.yaml')
@click.option('-r', '--rounds', required=False, default=1)
@click.option('-a', '--active', required=False, default=2)
@click.option('-t', '--timeout', required=False, default=120)
@click.option('-s', '--seedmodel', required=True)
def fedavg_cmd(ctx, rounds, active, timeout, seedmodel, config):
    import fedn.combiner.fedavg as fedavg
    # TODO override by input parameters
    config = {'round_timeout': timeout, 'seedmodel': seedmodel, 'rounds': rounds, 'active_clients': active}
    project = ctx.obj['PROJECT']

    # combiner = fedavg.Orchestrator(config=config)
    from fedn.combiner.helpers import get_combiner
    from fedn.combiner.server import FednServer
    server = FednServer(project, get_combiner)

    server.run(config)


@run_cmd.command('reducer')
@click.pass_context
def reducer_cmd(ctx, ):
    from fedn.reducer.reducer import Reducer
    project = ctx.obj['PROJECT']
    reducer = Reducer(project)

    reducer.run()
