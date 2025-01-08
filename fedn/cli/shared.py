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

home_dir = os.path.expanduser("~")


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
    _protocol = protocol or os.environ.get("FEDN_CONTROLLER_PROTOCOL") or CONTROLLER_DEFAULTS["protocol"]
    _host = host or os.environ.get("FEDN_CONTROLLER_HOST") or CONTROLLER_DEFAULTS["host"]
    _port = port or os.environ.get("FEDN_CONTROLLER_PORT") or CONTROLLER_DEFAULTS["port"]

    if _url is None:
        context_path = os.path.join(home_dir, ".fedn")
        try:
            with open(f"{context_path}/context.yaml", "r") as yaml_file:
                context_data = yaml.safe_load(yaml_file)
            _url = context_data.get("Active project url")
        except Exception as e:
            click.echo(f"Encountered error {e}. Make sure you are logged in and have activated a project. Using controller defaults instead.", fg="red")
            _url = f"{_protocol}://{_host}:{_port}"

    return f"{_url}/api/{API_VERSION}/{endpoint}/"


def get_token(token: str, usr_token: bool) -> str:
    _token = token or os.environ.get("FEDN_AUTH_TOKEN", None)

    if _token is None:
        context_path = os.path.join(home_dir, ".fedn")
        try:
            with open(f"{context_path}/context.yaml", "r") as yaml_file:
                context_data = yaml.safe_load(yaml_file)
            if usr_token:
                _token = context_data.get("User tokens").get("access")
            else:
                _token = context_data.get("Active project tokens").get("access")
        except Exception as e:
            click.echo(f"Encountered error {e}. Make sure you are logged in and have activated a project.", fg="red")

    scheme = os.environ.get("FEDN_AUTH_SCHEME", "Bearer")

    return f"{scheme} {_token}"


def get_client_package_dir(path: str) -> str:
    return path or os.environ.get("FEDN_PACKAGE_DIR", None)


# Print response from api (list of entities)
def print_response(response, entity_name: str, so):
    """Prints the api response to the cli.
    :param response:
        type: array
        description: list of entities
    :param entity_name:
        type: string
        description: name of entity
    :param so:
        type: boolean
        desriptions: single output format
    return: None
    """
    if response.status_code == 200:
        json_data = response.json()
        if so:
            click.echo(f"Found {entity_name}")
            click.echo("\n---------------------------------\n")
            for k, v in json_data.items():
                click.echo(f"\t{k}: {v}")
        else:
            count, result = json_data.values()
            click.echo(f"Found {count} {entity_name}")
            click.echo("\n---------------------------------\n")
            for obj in result:
                print(obj.get("session_id"))
                click.echo("{")
                for k, v in obj.items():
                    click.echo(f"\t{k}: {v}")
                click.echo("}")
    elif response.status_code == 500:
        json_data = response.json()
        click.echo(f'Error: {json_data["message"]}')
    else:
        click.echo(f"Error: {response.status_code}")
