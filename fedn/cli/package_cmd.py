"""Package commands for the CLI."""

import fnmatch
import os
import sys
import tarfile

import click

from fedn.cli.main import main
from fedn.cli.shared import CONTROLLER_DEFAULTS, get_response, print_response
from fedn.common.log_config import logger


def create_tar_with_ignore(path: str, output_path: str) -> None:
    """Create a tar archive from a directory with an ignore and fedn.yaml file."""
    try:
        ignore_patterns = []
        ignore_file = os.path.join(path, ".fednignore")
        if os.path.exists(ignore_file):
            # Read ignore patterns from .fednignore file
            with open(ignore_file, "r") as f:
                ignore_patterns = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        def is_ignored(file_path: str) -> bool:
            relative_path = os.path.relpath(file_path, path)
            return any(fnmatch.fnmatch(relative_path, pattern) or fnmatch.fnmatch(os.path.basename(file_path), pattern) for pattern in ignore_patterns)

        with tarfile.open(output_path, "w:gz") as tar:
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d))]
                for file in files:
                    file_path = os.path.join(root, file)
                    if not is_ignored(file_path):
                        logger.debug(f"Adding file to tar archive: {file_path}")
                        tar.add(file_path, arcname=os.path.relpath(file_path, path))

        logger.info(f"Created tar archive: {output_path}")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


@main.group("package")
@click.pass_context
def package_cmd(_: click.Context) -> None:
    """:param ctx:"""
    pass


@package_cmd.command("create")
@click.option("-p", "--path", required=True, help="Path to package directory containing fedn.yaml")
@click.option("-n", "--name", required=False, default="package.tgz", help="Name of package tarball")
@click.option("-o", "--output", required=False, default=os.getcwd(), help="Output directory for the generated tarball")
@click.pass_context
def create_cmd(_: click.Context, path: str, name: str, output: str) -> None:
    """Create compute package.

    Make a tar.gz archive of folder given by --path. The archive will be named --name and saved in --output.
    """
    try:
        path = os.path.abspath(path)
        output = os.path.abspath(output)
        yaml_file = os.path.join(path, "fedn.yaml")
        if not os.path.exists(yaml_file):
            logger.error(f"Could not find fedn.yaml in {path}")
            sys.exit(-1)

        if not os.path.exists(output):
            logger.error(f"Output directory does not exist: {output}")
            sys.exit(-1)

        tar_path = os.path.join(output, name)
        create_tar_with_ignore(path, tar_path)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(-1)


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("--n_max", required=False, help="Number of items to list")
@package_cmd.command("list")
@click.pass_context
def list_packages(_: click.Context, protocol: str, host: str, port: str, token: str = None, n_max: int = None) -> None:
    """Return a list of packages.

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
def get_package(_: click.Context, protocol: str, host: str, port: str, token: str = None, id: str = None) -> None:
    """Return a package with given id.

    ------
    - result: package with given id

    """
    response = get_response(protocol=protocol, host=host, port=port, endpoint=f"packages/{id}", token=token, headers={}, usr_api=False, usr_token=False)
    print_response(response, "package", id)
