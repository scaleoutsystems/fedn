import os

import click
import yaml

from fedn.common.log_config import logger

CONTROLLER_DEFAULTS = {"protocol": "http", "host": "localhost", "port": 8092, "debug": False}

COMBINER_DEFAULTS = {"discover_host": "localhost", "discover_port": 8092, "host": "localhost", "port": 12080, "name": "combiner", "max_clients": 30}

CLIENT_DEFAULTS = {
    "discover_host": "localhost",
    "discover_port": 8092,
}

API_VERSION = "v1"


def apply_config(path: str, config: dict):
    """Parse client config from file.

    Override configs from the CLI with settings in config file.

    :param config: Client config (dict).
    """
    with open(path, "r") as file:
        try:
            settings = dict(yaml.safe_load(file))
        except Exception:
            logger.error("Failed to read config from settings file, exiting.")
            return

    for key, val in settings.items():
        config[key] = val


def get_api_url(protocol: str, host: str, port: str, endpoint: str) -> str:
    _url = os.environ.get("FEDN_CONTROLLER_URL")

    if _url:
        return f"{_url}/api/{API_VERSION}/{endpoint}/"

    _protocol = protocol or os.environ.get("FEDN_CONTROLLER_PROTOCOL") or CONTROLLER_DEFAULTS["protocol"]
    _host = host or os.environ.get("FEDN_CONTROLLER_HOST") or CONTROLLER_DEFAULTS["host"]
    _port = port or os.environ.get("FEDN_CONTROLLER_PORT") or CONTROLLER_DEFAULTS["port"]

    return f"{_protocol}://{_host}:{_port}/api/{API_VERSION}/{endpoint}/"


def get_token(token: str) -> str:
    _token = token or os.environ.get("FEDN_AUTH_TOKEN", None)

    if _token is None:
        return None

    scheme = os.environ.get("FEDN_AUTH_SCHEME", "Bearer")

    return f"{scheme} {_token}"


def get_client_package_dir(path: str) -> str:
    return path or os.environ.get("FEDN_PACKAGE_DIR", None)


# Print response from api (list of entities)
def print_response(response, entity_name: str):
    """Prints the api response to the cli.
    :param response:
        type: array
        description: list of entities
    :param entity_name:
        type: string
        description: name of entity
    return: None
    """
    if response.status_code == 200:
        json_data = response.json()
        count, result = json_data.values()
        click.echo(f"Found {count} {entity_name}")
        click.echo("\n---------------------------------\n")
        for obj in result:
            click.echo("{")
            for k, v in obj.items():
                click.echo(f"\t{k}: {v}")
            click.echo("}")
    elif response.status_code == 500:
        json_data = response.json()
        click.echo(f'Error: {json_data["message"]}')
    else:
        click.echo(f"Error: {response.status_code}")
