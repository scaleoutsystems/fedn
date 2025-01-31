import os
import tarfile

import click

from fedn.common.log_config import logger

from .main import main
from .shared import CONTROLLER_DEFAULTS, get_response, print_response


@main.group("package")
@click.pass_context
def package_cmd(ctx):
    """:param ctx:"""
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
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    response = get_response(protocol=protocol, host=host, port=port, endpoint="packages/", token=token, headers=headers, usr_api=False, usr_token=False)
    print_response(response, "packages", None)


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-id", "--id", required=True, help="Package ID")
@package_cmd.command("get")
@click.pass_context
def get_package(ctx, protocol: str, host: str, port: str, token: str = None, id: str = None):
    """Return:
    ------
    - result: package with given id

    """
    response = get_response(protocol=protocol, host=host, port=port, endpoint=f"packages/{id}", token=token, headers={}, usr_api=False, usr_token=False)
    print_response(response, "package", id)
