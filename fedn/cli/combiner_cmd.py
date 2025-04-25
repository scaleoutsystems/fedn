import uuid

import click

from fedn.common.config import get_modelstorage_config, get_network_config, get_statestore_config
from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.s3.repository import Repository

from .main import main
from .shared import CONTROLLER_DEFAULTS, apply_config, get_response, print_response


@main.group("combiner")
@click.pass_context
def combiner_cmd(ctx):
    """:param ctx:"""
    pass


@combiner_cmd.command("start")
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

    if init:
        apply_config(init, config)
        click.echo(f"\nCombiner configuration loaded from file: {init}")
        click.echo("Values set in file override defaults and command line arguments...\n")

    from fedn.network.combiner.combiner import Combiner

    modelstorage_config = get_modelstorage_config()
    statestore_config = get_statestore_config()
    network_id = get_network_config()

    # TODO: set storage_type ?
    repository = Repository(modelstorage_config["storage_config"], init_buckets=False)

    db = DatabaseConnection(statestore_config, network_id)

    combiner = Combiner(config, repository, db)
    combiner.run()


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("--n_max", required=False, help="Number of items to list")
@combiner_cmd.command("list")
@click.pass_context
def list_combiners(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None):
    """Return:
    ------
    - count: number of combiners
    - result: list of combiners

    """
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    response = get_response(protocol=protocol, host=host, port=port, endpoint="combiners/", token=token, headers=headers, usr_api=False, usr_token=False)
    print_response(response, "combiners", None)


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-id", "--id", required=True, help="Combiner ID")
@combiner_cmd.command("get")
@click.pass_context
def get_combiner(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None):
    """Return:
    ------
    - result: combiner with given id

    """
    response = get_response(protocol=protocol, host=host, port=port, endpoint=f"combiners/{id}", token=token, headers={}, usr_api=False, usr_token=False)
    print_response(response, "combiner", id)
