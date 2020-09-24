import click

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
@click.option('-n', '--name', required=False, default=None)
@click.option('-i', '--client_id', required=False)
@click.pass_context
def client_cmd(ctx, discoverhost, discoverport, token, name, client_id):

    if not name:
        import uuid
        name = str(uuid.uuid4())
    
    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'name': name,
              'client_id': client_id}


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
@click.option('-s', '--secure', required=False, default=True)
@click.option('-c', '--max_clients', required=False,default=8)
@click.pass_context
def combiner_cmd(ctx, discoverhost, discoverport, token, name, hostname, port, secure,max_clients):
    config = {'discover_host': discoverhost, 'discover_port': discoverport, 'token': token, 'myhost': hostname,
              'myport': port, 'myname': name, 'secure': secure,'max_clients':max_clients}

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
