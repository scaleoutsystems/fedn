import os
import tarfile
import click
import logging

from fedn.common.log_config import logger

from fedn.cli.main import main
from fedn.cli.shared import CONTROLLER_DEFAULTS, get_response, print_response

def create_tar_with_ignore(path, name):
    try:
        ignore_patterns = []
        ignore_file = os.path.join(path, ".ignore")
        if os.path.exists(ignore_file):
            # Read ignore patterns from .ignore file
            with open(ignore_file, 'r') as f:
                ignore_patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        def is_ignored(file_path):
            for pattern in ignore_patterns:
                if pattern in file_path:
                    return True
            return False

        tar_path = os.path.join(path, name)
        with tarfile.open(tar_path, "w:gz") as tar:
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if not is_ignored(file_path):
                        logger.info(f"Adding file to tar archive: {file_path}")
                        tar.add(file_path, arcname=os.path.relpath(file_path, path))
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    if is_ignored(dir_path):
                        dirs.remove(dir)

        logger.info(f"Created tar archive: {tar_path}")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

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
    :param name:
    """
    try:
        path = os.path.abspath(path)
        yaml_file = os.path.join(path, "fedn.yaml")
        if not os.path.exists(yaml_file):
            logger.error(f"Could not find fedn.yaml in {path}")
            exit(-1)

        create_tar_with_ignore(path, name)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        exit(-1)

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
