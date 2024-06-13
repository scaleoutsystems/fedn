import uuid

import click
import requests

from fedn.network.combiner.combiner import Combiner

from .main import main
from .shared import CONTROLLER_DEFAULTS, apply_config, get_api_url, get_token, print_response


@main.group("hooks")
@click.pass_context
def hooks_cmd(ctx):
    """:param ctx:"""
    pass


@hooks_cmd.command("start")
@click.option("-d", "--discoverhost", required=False, help="Hostname for discovery services (reducer).")
@click.option("-p", "--discoverport", required=False, help="Port for discovery services (reducer).")
@click.option("-t", "--token", required=False, help="Set token provided by reducer if enabled")
@click.option("-n", "--name", required=False, default="combiner" + str(uuid.uuid4())[:8], help="Set name for combiner.")
@click.option("-h", "--host", required=False, default="combiner", help="Set hostname.")
@click.option("-i", "--port", required=False, default=12080, help="Set port.")
@click.option("-f", "--fqdn", required=False, default=None, help="Set fully qualified domain name")
@click.option("-s", "--secure", is_flag=True, help="Enable SSL/TLS encrypted gRPC channels.")
@click.option("-v", "--verify", is_flag=True, help="Verify SSL/TLS for REST discovery service (reducer)")
@click.option("-c", "--max_clients", required=False, default=30, help="The maximal number of client connections allowed.")
@click.option("-in", "--init", required=False, default=None, help="Path to configuration file to (re)init combiner.")
@click.pass_context
def start_cmd(ctx, discoverhost, discoverport, token, name, host, port, fqdn, secure, verify, max_clients, init):
    """:param ctx:
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
    config = {
        "discover_host": discoverhost,
        "discover_port": discoverport,
        "token": token,
        "host": host,
        "port": port,
        "fqdn": fqdn,
        "name": name,
        "secure": secure,
        "verify": verify,
        "max_clients": max_clients,
    }
    click.echo("started hooks container")
    if init:
        apply_config(init, config)
        click.echo(f"\nCombiner configuration loaded from file: {init}")
        click.echo("Values set in file override defaults and command line arguments...\n")

    # combiner = Combiner(config)
    # combiner.run()
