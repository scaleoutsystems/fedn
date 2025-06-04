"""Package commands for the CLI."""

import fnmatch
import os
import sys
import tarfile

import click
import requests

from fedn.cli.main import main
from fedn.common.log_config import logger

from .shared import CONTROLLER_DEFAULTS, get_api_url, get_response, get_token, print_response


def create_tar_with_ignore(path: str, name: str) -> None:
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

        tar_path = os.path.join(path, name)
        with tarfile.open(tar_path, "w:gz") as tar:
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d))]
                for file in files:
                    file_path = os.path.join(root, file)
                    if not is_ignored(file_path):
                        logger.debug(f"Adding file to tar archive: {file_path}")
                        tar.add(file_path, arcname=os.path.relpath(file_path, path))

        logger.info(f"Created tar archive: {tar_path}")
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
@click.pass_context
def create_cmd(_: click.Context, path: str, name: str) -> None:
    """Create compute package.

    Make a tar.gz archive of folder given by --path. The archive will be named --name.
    """
    try:
        path = os.path.abspath(path)
        yaml_file = os.path.join(path, "fedn.yaml")
        if not os.path.exists(yaml_file):
            logger.error(f"Could not find fedn.yaml in {path}")
            sys.exit(-1)

        create_tar_with_ignore(path, name)
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


@click.option("-p", "--protocol", required=False, default=CONTROLLER_DEFAULTS["protocol"], help="Communication protocol of controller (api)")
@click.option("-H", "--host", required=False, default=CONTROLLER_DEFAULTS["host"], help="Hostname of controller (api)")
@click.option("-P", "--port", required=False, default=CONTROLLER_DEFAULTS["port"], help="Port of controller (api)")
@click.option("-t", "--token", required=False, help="Authentication token")
@click.option("-f", "--file", required=True, help="Path to the compute package file")
@package_cmd.command("upload")
@click.pass_context
def set_active_package(ctx, protocol: str, host: str, port: str, token: str, file: str):
    """Uploads compute package to package repository."""
    headers = {}
    _token = get_token(token=token, usr_token=False)
    if _token:
        headers = {"Authorization": _token}

    if file.endswith(".tgz"):
        helper = "numpyhelper"
    elif file.endswith(".bin"):
        helper = "binaryhelper"
    else:
        click.secho("Unsupported file type. Only .tgz and .bin files are supported.", fg="red")
        return

    try:
        # Set the active helper
        url = get_api_url(protocol, host, port, "helpers/active", usr_api=False)
        response = requests.put(url, json={"helper": helper}, headers=headers, verify=False)
        response.raise_for_status()

        # Upload the model file
        url = get_api_url(protocol, host, port, "package/", usr_api=False)
        with open(file, "rb") as package_file:
            response = requests.post(url, files={"file": package_file}, data={"helper": helper}, headers=headers, verify=False)
            response.raise_for_status()

        click.secho("Package set as active and uploaded successfully.", fg="green")
    except requests.exceptions.RequestException as e:
        click.secho(f"Failed to set active package: {e}", fg="red")
