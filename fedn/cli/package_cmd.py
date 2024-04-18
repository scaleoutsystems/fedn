import os
import tarfile

import click

from fedn.common.log_config import logger

from .main import main


@main.group('package')
@click.pass_context
def package_cmd(ctx):
    """

    :param ctx:
    """
    pass


@package_cmd.command('create')
@click.option('-p', '--path', required=True, help='Path to package directory containing fedn.yaml')
@click.option('-n', '--name', required=False, default='package.tgz', help='Name of package tarball')
@click.pass_context
def create_cmd(ctx, path, name):
    """ Create compute package.

    Make a tar.gz archive of folder given by --path

    :param ctx:
    :param path:
    """
    path = os.path.abspath(path)
    yaml_file = os.path.join(path, 'fedn.yaml')
    if not os.path.exists(yaml_file):
        logger.error(f"Could not find fedn.yaml in {path}")
        exit(-1)

    with tarfile.open(name, "w:gz") as tar:
        tar.add(path, arcname=os.path.basename(path))
        logger.info(f"Created package {name}")