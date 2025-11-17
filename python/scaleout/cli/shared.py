import os
from typing import Optional, Tuple

import click
from scaleoututil.utils.url import build_url, assemble_endpoint_url, parse_url
import requests
import yaml

from scaleoututil.logging import FednLogger

CONTROLLER_DEFAULTS = {"local_protocol": "http", "protocol": "https", "host": "localhost", "port": 8092, "debug": False}

HOME_DIR = os.environ.get("SCALEOUT_HOME_DIR", os.path.expanduser("~"))

API_VERSION = "v1"

CONTEXT_FOLDER = ".scaleout"


def apply_config(path: str, config: dict):
    """Parse client config from file.

    Override configs from the CLI with settings in config file.

    :param config: Client config (dict).
    """
    with open(path, "r") as file:
        try:
            settings = dict(yaml.safe_load(file))
        except Exception:
            FednLogger().error("Failed to read config from settings file, exiting.")
            return

    for key, val in settings.items():
        config[key] = val


def get_api_url(base_url: str, endpoint: str, query: dict = None) -> str:
    """Utility function to build api url based on provided information."""
    return assemble_endpoint_url(base_url, f"api/{API_VERSION}", endpoint, **(query or {}))


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
        if json_data.get("count") is not None and json_data.get("result") is not None:
            # list output
            count = json_data.get("count")
            result = json_data.get("result")
            click.echo(f"Found {count} {entity_name}")
            click.echo("\n---------------------------------\n")
            for obj in result:
                click.echo("{")
                for k, v in obj.items():
                    click.echo(f"\t{k}: {v}")
                click.echo("}")
        else:
            click.echo(f"Found {entity_name}")
            click.echo("\n---------------------------------\n")
            for k, v in json_data.items():
                click.echo(f"\t{k}: {v}")
    elif response.status_code == 500:
        json_data = response.json()
        click.secho(f"Error: {json_data['message']}", fg="red")
    else:
        click.secho(f"Error: {response.status_code}", fg="red")


def set_context(host, token):
    """Saves context data as yaml file in given path"""
    context_data = {
        "host": host,
        "token": token,
    }
    try:
        if not os.path.exists(f"{HOME_DIR}/{CONTEXT_FOLDER}"):
            os.makedirs(f"{HOME_DIR}/{CONTEXT_FOLDER}")
        with open(f"{HOME_DIR}/{CONTEXT_FOLDER}/context.yaml", "w") as yaml_file:
            yaml.dump(context_data, yaml_file, default_flow_style=False)
    except Exception as e:
        print(f"Error: Failed to write to YAML file. Details: {e}")


def get_context():
    """Retrieves context data from yaml file in given path"""
    try:
        with open(f"{HOME_DIR}/{CONTEXT_FOLDER}/context.yaml", "r") as yaml_file:
            context_data = yaml.safe_load(yaml_file)
    except Exception as e:
        print(f"Error: Failed to read YAML file. Details: {e}")
        context_data = {}
    return context_data


def get_response(base_url: str, endpoint: str, query: dict, token: str, headers: dict):
    """Utility function to retrieve response from get request based on provided information."""
    url = get_api_url(base_url=base_url, endpoint=endpoint, query=query)
    _token = get_scheme_token(token)
    if _token:
        headers["Authorization"] = _token

    response = requests.get(url, headers=headers)

    return response


def complement_with_context(protocol: Optional[str], host: Optional[str], port: Optional[str], token: Optional[str]) -> Tuple[str, Optional[str]]:
    """Utility function to retrieve host from context file."""
    if not host:
        if protocol or port:
            raise ValueError("Both protocol and port must be provided together with host.")
        context_path = os.path.join(HOME_DIR, CONTEXT_FOLDER, "context.yaml")
        if os.path.exists(context_path):
            context_data = get_context(context_path)
            host = context_data.get("host")
            FednLogger().info(f"Using host: '{host}' from context file {context_path}")
        else:
            context_data = {}
            host = None

        # TODO: which prioritizes env vs context file?
        token = token or context_data.get("token") or os.environ.get("SCALEOUT_AUTH_TOKEN")

    host = complement_with_defaults(protocol, host, port)
    return host, token


def complement_with_defaults(protocol: Optional[str], host: Optional[str], port: Optional[str]) -> str:
    """Utility function to build host URL with defaults when necessary."""
    if host:
        _protocol, _host, _port, _ = parse_url(host)
        protocol = protocol or _protocol
        port = port or _port
        if port is None:
            port = CONTROLLER_DEFAULTS["port"]
            FednLogger().info("Complementing port with default: " + str(CONTROLLER_DEFAULTS["port"]))
        if _host in ["localhost", "127.0.0.1"]:
            if protocol is None:
                protocol = CONTROLLER_DEFAULTS["local_protocol"]
                FednLogger().info("Complementing protocol with default: " + CONTROLLER_DEFAULTS["local_protocol"])
        elif protocol is None:
            protocol = CONTROLLER_DEFAULTS["protocol"]
            FednLogger().info("Complementing protocol with default: " + CONTROLLER_DEFAULTS["protocol"])
        host = _host
        host = build_url(protocol, host, port, "")
    else:
        if protocol or port:
            raise ValueError("Both protocol and port must be provided together with host.")
        FednLogger().info("Using default controller host settings")
        host = build_url(CONTROLLER_DEFAULTS["local_protocol"], CONTROLLER_DEFAULTS["host"], CONTROLLER_DEFAULTS["port"], "")
    return host


def get_scheme_token(token: str):
    if token:
        token_scheme = os.environ.get("SCALEOUT_AUTH_SCHEME", "Bearer")
        token = f"{token_scheme} {token}"
    return token
