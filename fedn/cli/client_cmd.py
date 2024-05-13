import uuid

import click
import requests

from fedn.common.exceptions import InvalidClientConfig
from fedn.network.clients.client import Client

from .main import main
from .shared import CONTROLLER_DEFAULTS, apply_config, get_api_url, get_token, print_response


def validate_client_config(config):
    """Validate client configuration.

    :param config: Client config (dict).
    """
    try:
        if config["discover_host"] is None or config["discover_host"] == "":
            raise InvalidClientConfig("Missing required configuration: discover_host")
        if "discover_port" not in config.keys():
            config["discover_port"] = None
    except Exception:
        raise InvalidClientConfig("Could not load config from file. Check config")


@main.group("client")
@click.pass_context
def client_cmd(ctx):
    """:param ctx:
    """
    pass


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("--n_max", required=False, help="Number of items to list")
@client_cmd.command("list")
@click.pass_context
def list_clients(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None):
    """Return:
    ------
    - count: number of clients
    - result: list of clients

    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint="clients")
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    _token = get_token(token)

    if _token:
        headers["Authorization"] = _token

    click.echo(f"\nListing clients: {url}\n")
    click.echo(f"Headers: {headers}")

    try:
        response = requests.get(url, headers=headers)
        print_response(response, "clients")
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url}")


@client_cmd.command("start")
@click.option("-d", "--discoverhost", required=False, help="Hostname for discovery services(reducer).")
@click.option("-p", "--discoverport", required=False, help="Port for discovery services (reducer).")
@click.option("--token", required=False, help="Set token provided by reducer if enabled")
@click.option("-n", "--name", required=False, default="client" + str(uuid.uuid4())[:8])
@click.option("-i", "--client_id", required=False)
@click.option("--local-package", is_flag=True, help="Enable local compute package")
@click.option("--force-ssl", is_flag=True, help="Force SSL/TLS for REST service")
@click.option("-u", "--dry-run", required=False, default=False)
@click.option("-s", "--secure", required=False, default=False)
@click.option("-pc", "--preshared-cert", required=False, default=False)
@click.option("-v", "--verify", is_flag=True, help="Verify SSL/TLS for REST service")
@click.option("-c", "--preferred-combiner", required=False, default=False)
@click.option("-va", "--validator", required=False, default=True)
@click.option("-tr", "--trainer", required=False, default=True)
@click.option("-in", "--init", required=False, default=None, help="Set to a filename to (re)init client from file state.")
@click.option("-l", "--logfile", required=False, default=None, help="Set logfile for client log to file.")
@click.option("--heartbeat-interval", required=False, default=2)
@click.option("--reconnect-after-missed-heartbeat", required=False, default=30)
@click.option("--verbosity", required=False, default="INFO", type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], case_sensitive=False))
@click.pass_context
def client_cmd(
    ctx,
    discoverhost,
    discoverport,
    token,
    name,
    client_id,
    local_package,
    force_ssl,
    dry_run,
    secure,
    preshared_cert,
    verify,
    preferred_combiner,
    validator,
    trainer,
    init,
    logfile,
    heartbeat_interval,
    reconnect_after_missed_heartbeat,
    verbosity,
):
    """:param ctx:
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
    :param verbosity
    :return:
    """
    remote = False if local_package else True
    config = {
        "discover_host": discoverhost,
        "discover_port": discoverport,
        "token": token,
        "name": name,
        "client_id": client_id,
        "remote_compute_context": remote,
        "force_ssl": force_ssl,
        "dry_run": dry_run,
        "secure": secure,
        "preshared_cert": preshared_cert,
        "verify": verify,
        "preferred_combiner": preferred_combiner,
        "validator": validator,
        "trainer": trainer,
        "logfile": logfile,
        "heartbeat_interval": heartbeat_interval,
        "reconnect_after_missed_heartbeat": reconnect_after_missed_heartbeat,
        "verbosity": verbosity,
    }

    if init:
        apply_config(init, config)
        click.echo(f"\nClient configuration loaded from file: {init}")
        click.echo("Values set in file override defaults and command line arguments...\n")

    try:
        validate_client_config(config)
    except InvalidClientConfig as e:
        click.echo(f"Error: {e}")
        return

    client = Client(config)
    client.run()
