import os
import tarfile

import click
import requests

from fedn.common.log_config import logger

from .main import main
from .shared import CONTROLLER_DEFAULTS, get_api_url, get_token, print_response


@main.group("package")
@click.pass_context
def package_cmd(ctx):
    """:param ctx:
    """
    pass


@package_cmd.command("create")
@click.option("-p", "--path", required=True, help="Path to package directory containing fedn.yaml")
@click.option("-n", "--name", required=False, default="package.tgz", help="Name of package tarball")
@click.pass_context
def create_cmd(ctx, path, name):
    """Create compute package.

    Make a tar.gz archive of folder given by --path

    :param ctx:
    :param path:
    """
    path = os.path.abspath(path)
    yaml_file = os.path.join(path, "fedn.yaml")
    if not os.path.exists(yaml_file):
        logger.error(f"Could not find fedn.yaml in {path}")
        exit(-1)

    with tarfile.open(name, "w:gz") as tar:
        tar.add(path, arcname=os.path.basename(path))
        logger.info(f"Created package {name}")


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("--n_max", required=False, help="Number of items to list")
@package_cmd.command("list")
@click.pass_context
def list_packages(ctx, protocol: str, host: str, port: str, token: str = None, n_max: int = None):
    """Return:
    ------
    - count: number of packages
    - result: list of packages

    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint="packages")
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    _token = get_token(token)

    if _token:
        headers["Authorization"] = _token

    click.echo(f"\nListing packages: {url}\n")
    click.echo(f"Headers: {headers}")

    try:
        response = requests.get(url, headers=headers)
        print_response(response, "packages")
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url}")
