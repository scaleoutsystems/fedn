import uuid

import click

from fedn.cli.main import main
from fedn.cli.shared import CONTROLLER_DEFAULTS, apply_config, get_response, print_response
from fedn.common.exceptions import InvalidClientConfig
from fedn.network.clients.client_v2 import Client as ClientV2
from fedn.network.clients.client_v2 import ClientOptions


def validate_client_config(config):
    """Validate client configuration.

    :param config: Client config (dict).
    """
    try:
        if config["discover_host"] is None or config["discover_host"] == "":
            if config["combiner"] is None or config["combiner"] == "":
                raise InvalidClientConfig("Missing required configuration: discover_host or combiner")
        if "discover_port" not in config.keys():
            config["discover_port"] = None
        if config["remote_compute_context"] and config["discover_host"] is None:
            raise InvalidClientConfig("Remote compute context requires discover_host")
    except Exception:
        raise InvalidClientConfig("Could not load config from file. Check config")


@main.group("client")
@click.pass_context
def client_cmd(ctx):
    """- Commands for listing and running clients."""
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
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    response = get_response(protocol=protocol, host=host, port=port, endpoint="clients", token=token, headers=headers, usr_api=False, usr_token=False)
    print_response(response, "clients", None)


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-id", "--id", required=True, help="Client ID")
@client_cmd.command("get")
@click.pass_context
def get_client(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None):
    """Return:
    ------
    - result: client with given id

    """
    response = get_response(protocol=protocol, host=host, port=port, endpoint=f"clients/{id}", token=token, headers={}, usr_api=False, usr_token=False)
    print_response(response, "client", id)


def _validate_client_params(config: dict):
    api_url = config["api_url"]
    combiner = config["combiner"]
    combiner_port = config["combiner_port"]
    if (api_url is None or api_url == "") and (combiner is None or combiner == ""):
        click.echo("Error: Missing required parameter: --api_url or --combiner")
        return False
    if (combiner is not None and combiner != "") and (combiner_port is None or combiner_port == ""):
        click.echo("Error: Missing required parameter: --combiner-port")
        return False
    return True


def _complement_client_params(config: dict):
    api_url = config["api_url"]
    if not api_url.startswith("http://") and not api_url.startswith("https://"):
        if "localhost" in api_url or "127.0.0.1" in api_url:
            config["api_url"] = "http://" + api_url
        else:
            config["api_url"] = "https://" + api_url
        result = config["api_url"]

        click.echo(f"Protocol missing, complementing api_url with protocol: {result}")


@client_cmd.command("start")
@click.option("-u", "--api-url", required=False, help="Hostname for fedn api.")
@click.option("-p", "--api-port", required=False, help="Port for discovery services (reducer).")
@click.option("--token", required=False, help="Set token provided by reducer if enabled")
@click.option("-n", "--name", required=False)
@click.option("-i", "--client-id", required=False)
@click.option("--local-package", is_flag=True, help="Enable local compute package")
@click.option("-c", "--preferred-combiner", type=str, required=False, default="", help="name of the preferred combiner")
@click.option("--combiner", type=str, required=False, default=None, help="Skip combiner assignment from discover service and attach directly to combiner host.")
@click.option("--combiner-port", type=str, required=False, default=None, help="Combiner port, need to be used with --combiner")
@click.option("-va", "--validator", required=False, default=None)
@click.option("-tr", "--trainer", required=False, default=None)
@click.option("-h", "--helper_type", required=False, default=None)
@click.option("-in", "--init", required=False, default=None, help="Set to a filename to (re)init client from file state.")
@click.pass_context
def client_start_v2_cmd(
    ctx,
    api_url: str,
    api_port: int,
    token: str,
    name: str,
    client_id: str,
    local_package: bool,
    preferred_combiner: str,
    combiner: str,
    combiner_port: int,
    validator: bool,
    trainer: bool,
    helper_type: str,
    init: str,
):
    package = "local" if local_package else "remote"

    config = {
        "api_url": None,
        "api_port": None,
        "token": None,
        "name": None,
        "client_id": None,
        "preferred_combiner": None,
        "combiner": None,
        "combiner_port": None,
        "validator": None,
        "trainer": None,
        "package_checksum": None,
        "helper_type": None,
        # to cater for old inputs
        "discover_host": None,
        "discover_port": None,
    }

    if init:
        apply_config(init, config)
        click.echo(f"\nClient configuration loaded from file: {init}")

        # to cater for old inputs
        if config["discover_host"] is not None:
            config["api_url"] = config["discover_host"]

        if config["discover_port"] is not None:
            config["api_port"] = config["discover_port"]

    if api_url and api_url != "":
        config["api_url"] = api_url
        if config["api_url"] and config["api_url"] != "":
            click.echo(f"Input param api_url: {api_url} overrides value from file")

    if api_port:
        config["api_port"] = api_port
        if config["api_port"]:
            click.echo(f"Input param api_port: {api_port} overrides value from file")

    if token and token != "":
        config["token"] = token
        if config["token"]:
            click.echo(f"Input param token: {token} overrides value from file")

    if name and name != "":
        config["name"] = name
        if config["name"]:
            click.echo(f"Input param name: {name} overrides value from file")
    elif config["name"] is None:
        config["name"] = "client" + str(uuid.uuid4())[:8]

    if client_id and client_id != "":
        config["client_id"] = client_id
        if config["client_id"]:
            click.echo(f"Input param client_id: {client_id} overrides value from file")

    if preferred_combiner and preferred_combiner != "":
        config["preferred_combiner"] = preferred_combiner
        if config["preferred_combiner"]:
            click.echo(f"Input param preferred_combiner: {preferred_combiner} overrides value from file")

    if combiner and combiner != "":
        config["combiner"] = combiner
        if config["combiner"]:
            click.echo(f"Input param combiner: {combiner} overrides value from file")

    if combiner_port:
        config["combiner_port"] = combiner_port
        if config["combiner_port"]:
            click.echo(f"Input param combiner_port: {combiner_port} overrides value from file")

    if validator is not None:
        config["validator"] = validator
        if config["validator"] is not None:
            click.echo(f"Input param validator: {validator} overrides value from file")
    elif config["validator"] is None:
        config["validator"] = True

    if trainer is not None:
        config["trainer"] = trainer
        if config["trainer"] is not None:
            click.echo(f"Input param trainer: {trainer} overrides value from file")
    elif config["trainer"] is None:
        config["trainer"] = True

    if helper_type and helper_type != "":
        config["helper_type"] = helper_type
        if config["helper_type"]:
            click.echo(f"Input param helper_type: {helper_type} overrides value from file")

    if not _validate_client_params(config):
        return

    if config["api_url"] is not None and config["api_url"] != "":
        _complement_client_params(config)

    client_options = ClientOptions(
        name=config["name"],
        package=package,
        preferred_combiner=config["preferred_combiner"],
        id=config["client_id"],
    )
    client = ClientV2(
        api_url=config["api_url"],
        api_port=config["api_port"],
        client_obj=client_options,
        combiner_host=config["combiner"],
        combiner_port=config["combiner_port"],
        token=config["token"],
        package_checksum=config["package_checksum"],
        helper_type=config["helper_type"],
    )

    client.start()
