import os
import shutil

import click
import yaml

from fedn.common.log_config import logger
from fedn.utils.dispatcher import Dispatcher, _read_yaml_file

from .main import main


def get_statestore_config_from_file(init):
    """

    :param init:
    :return:
    """
    with open(init, 'r') as file:
        try:
            settings = dict(yaml.safe_load(file))
            return settings
        except yaml.YAMLError as e:
            raise (e)


def check_helper_config_file(config):
    control = config['control']
    try:
        helper = control["helper"]
    except KeyError:
        logger.error("--local-package was used, but no helper was found in --init settings file.")
        exit(-1)
    return helper


@main.group('run')
@click.pass_context
def run_cmd(ctx):
    """

    :param ctx:
    """
    pass


@run_cmd.command('build')
@click.option('-p', '--path', required=True, help='Path to package directory containing fedn.yaml')
@click.pass_context
def build_cmd(ctx, path):
    """ Execute 'build' entrypoint in fedn.yaml.

    :param ctx:
    :param path: Path to folder containing fedn.yaml
    :type path: str
    """
    path = os.path.abspath(path)
    yaml_file = os.path.join(path, 'fedn.yaml')
    if not os.path.exists(yaml_file):
        logger.error(f"Could not find fedn.yaml in {path}")
        exit(-1)

    config = _read_yaml_file(yaml_file)
    # Check that build is defined in fedn.yaml under entry_points
    if 'build' not in config['entry_points']:
        logger.error("No build command defined in fedn.yaml")
        exit(-1)

    dispatcher = Dispatcher(config, path)
    _ = dispatcher._get_or_create_python_env()
    dispatcher.run_cmd("build")

    # delete the virtualenv
    if dispatcher.python_env_path:
        logger.info(f"Removing virtualenv {dispatcher.python_env_path}")
        shutil.rmtree(dispatcher.python_env_path)
