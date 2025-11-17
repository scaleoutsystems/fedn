"""Client commands for the CLI."""

import os
import uuid

import click
from scaleoututil.utils.url import build_url, parse_url
import yaml

from scaleout.cli.main import main
from scaleout.cli.shared import apply_config, get_response, print_response, complement_with_context
from scaleoututil.logging import FednLogger
from scaleout.client.connect import ClientOptions
from scaleout.client.dispatcher_client import DispatcherClient
from scaleout.client.importer_client import ImporterClient

home_dir = os.path.expanduser("~")

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


@main.group(
    "client",
    help="Commands to generate client configs, list or fetch clients, and start a client instance.",
    invoke_without_command=True,
)
@click.pass_context
def client_cmd(ctx):
    """Client commands."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@click.option("-p", "--path", required=False, help="Path to where client yaml file will be located")
@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of api-server (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of api-server (api)")
@click.option("-P", "--port", required=False, default=None, type=int, help="Port of api-server (api)")
@click.option("-t", "--token", required=False, help="Authentication admin token")
@click.option("-g", "--group", required=False, default=1, help="number of clients to generate in a bulk")
@click.option("-n", "--name", required=True, help="Client name, will be used as prefix for client names. The client name will be suffixed with an index.")
@client_cmd.command("get-config")
@click.pass_context
def create_client(ctx, path: str, protocol: str, host: str, port: int, token: str = None, name: str = None, group: int = None):
    """Generate client config file(s).
    ------
    client config file(s) with following content:
    client_id: uuid
    discover_host: controller, get from context file
    name: client name (set prefix with options)
    refresh_token: unique refresh token for client
    token: unique access token for client

    """
    discover_host, admin_token = complement_with_context(protocol, host, port, token)

    # Ensure the target directory exists
    try:
        if not path:
            path = os.getcwd()
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            click.echo(f"Path does not exist: {abs_path}")
            click.echo(f"Creating path: {abs_path}")
            os.makedirs(abs_path)
    except PermissionError as e:
        click.echo(f"Error: Permission denied. Details: {e}", fg="red")

    for i in range(group):
        client_id = str(uuid.uuid4())

        # TODO: Fill in with response from token issuer
        response_json = {}

        client_data = {
            "client_id": client_id,
            "discover_host": discover_host,
            "name": f"{name}_{i}",
            "refresh_token": response_json.get("refresh"),
            "token": response_json.get("access"),
        }
        click.echo(f"{i}: Generating client config: {client_data}")
        try:
            client_yaml_path = os.path.join(abs_path, f"{name}_{i}.yaml")
            with open(client_yaml_path, "w") as yaml_file:
                yaml.dump(client_data, yaml_file, default_flow_style=False)
                click.echo(f"{i}: Client config file saved to: {client_yaml_path}")
        except PermissionError as e:
            click.echo(f"Error: Permission denied. Details: {e}", fg="red")
        except Exception as e:
            print(f"Error: Failed to write to YAML file. Details: {e}", fg="red")


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("--n_max", required=False, help="Number of items to list")
@client_cmd.command("list")
@click.pass_context
def list_clients(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None):
    """List clients.
    ------
    - count: number of clients
    - result: list of clients
    """
    base_url, token = complement_with_context(protocol, host, port, token)
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    response = get_response(base_url=base_url, endpoint="clients/", query={}, token=token, headers=headers)
    print_response(response, "clients")


@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=None, help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=None, help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-id", "--id", required=True, help="Client ID")
@client_cmd.command("get")
@click.pass_context
def get_client(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None):
    """Get client.
    ------
    - result: client with given id
    """
    base_url, token = complement_with_context(protocol, host, port, token)
    response = get_response(base_url=base_url, endpoint=f"clients/{id}", query={}, token=token, headers={})
    print_response(response, "client")


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


def _complement_client_params(config: dict) -> None:
    """Ensures that the 'api_url' in the provided configuration dictionary has a protocol (http or https).

    If the 'api_url' does not start with 'http://' or 'https://', it will prepend 'http://' if the URL contains
    'localhost' or '127.0.0.1'. Otherwise, it will prepend 'https://'.
    """
    api_url = config["api_url"]
    scheme, host, port, path = parse_url(api_url)
    port = config.get("api_port") or port

    if scheme is None:
        if host in ["localhost", "127.0.0.1"]:
            scheme = "http"
        else:
            scheme = "https"

        result = build_url(scheme, host, port, path)
        config["api_url"] = result
        click.echo(f"Protocol missing, complementing api_url with protocol: {result}")


@client_cmd.command("start")
@click.option("-u", "--api-url", required=False, help="Hostname for scaleout api.")
@click.option("-p", "--api-port", required=False, help="Port for discovery services (reducer).")
@click.option("--token", required=False, help="Set token provided by reducer if enabled")
@click.option("-n", "--name", required=False)
@click.option("-i", "--client-id", required=False)
@click.option("--local-package", is_flag=True, help="Enable local compute package")
@click.option(
    "--log-level",
    required=False,
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
@click.option("-c", "--preferred-combiner", type=str, required=False, default="", help="name of the preferred combiner")
@click.option("--combiner", type=str, required=False, default=None, help="Skip combiner assignment from discover service and attach directly to combiner host.")
@click.option("--combiner-port", type=str, required=False, default=None, help="Combiner port, need to be used with --combiner")
@click.option("-va", "--validator", required=False, default=None)
@click.option("-tr", "--trainer", required=False, default=None)
@click.option("-hp", "--helper_type", required=False, default=None)
@click.option("-in", "--init", required=False, default=None, help="Set to a filename to (re)init client from file state.")
@click.option("--importer", is_flag=True, help="Use the importer client instead of the dispatcher client.")
@click.option("--managed-env", is_flag=True, help="Use the managed environment for the client. Requires an active virtual environment residing in cwd.")
@click.pass_context
def client_start_v2_cmd(
    ctx,
    api_url: str,
    api_port: int,
    token: str,
    name: str,
    client_id: str,
    local_package: bool,
    log_level: str,
    preferred_combiner: str,
    combiner: str,
    combiner_port: int,
    validator: bool,
    trainer: bool,
    helper_type: str,
    init: str,
    importer: bool,
    managed_env: bool = False,
):
    """Start client."""
    package = "local" if local_package else "remote"

    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        click.echo(f"Invalid log level: {log_level}. Defaulting to INFO.")
        log_level = "INFO"

    FednLogger().set_log_level_from_string(log_level)

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
        click.echo(f"Client configuration loaded from file: {init}")

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
            click.echo("Input param token overrides value from file")

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

    if config["api_url"]:
        _complement_client_params(config)

    client_options = ClientOptions(
        name=config["name"],
        package=package,
        preferred_combiner=config["preferred_combiner"],
        client_id=config["client_id"],
    )
    if importer:
        client = ImporterClient(
            api_url=config["api_url"],
            client_obj=client_options,
            combiner_host=config["combiner"],
            combiner_port=config["combiner_port"],
            token=config["token"],
            package_checksum=config["package_checksum"],
            helper_type=config["helper_type"],
            managed_env=managed_env,
        )
    else:
        client = DispatcherClient(
            api_url=config["api_url"],
            client_obj=client_options,
            combiner_host=config["combiner"],
            combiner_port=config["combiner_port"],
            token=config["token"],
            package_checksum=config["package_checksum"],
            helper_type=config["helper_type"],
        )

    client.start()
