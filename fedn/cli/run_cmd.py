import os
import shutil
import uuid

import click
import yaml

from fedn.cli.client_cmd import validate_client_config
from fedn.cli.main import main
from fedn.cli.shared import apply_config
from fedn.common.exceptions import InvalidClientConfig
from fedn.common.log_config import logger
from fedn.network.clients.client import Client
from fedn.utils.dispatcher import Dispatcher, _read_yaml_file


def get_statestore_config_from_file(init):
    """:param init:
    :return:
    """
    with open(init, "r") as file:
        try:
            settings = dict(yaml.safe_load(file))
            return settings
        except yaml.YAMLError as e:
            raise (e)


def check_helper_config_file(config):
    control = config["control"]
    try:
        helper = control["helper"]
    except KeyError:
        logger.error("--local-package was used, but no helper was found in --init settings file.")
        exit(-1)
    return helper


def check_yaml_exists(path):
    """Check if fedn.yaml exists in the given path."""
    yaml_file = os.path.join(path, "fedn.yaml")
    if not os.path.exists(yaml_file):
        logger.error(f"Could not find fedn.yaml in {path}")
        click.echo(f"Could not find fedn.yaml in {path}")
        exit(-1)
    return yaml_file


def delete_virtual_environment(dispatcher):
    if dispatcher.python_env_path:
        logger.info(f"Removing virtualenv {dispatcher.python_env_path}")
        shutil.rmtree(dispatcher.python_env_path)
    else:
        logger.warning("No virtualenv found to remove.")


@main.group("run")
@click.pass_context
def run_cmd(ctx):
    """:param ctx:"""
    pass


@run_cmd.command("validate")
@click.option("-p", "--path", required=True, help="Path to package directory containing fedn.yaml")
@click.option("-i", "--input", required=True, help="Path to input model")
@click.option("-o", "--output", required=True, help="Path to write the output JSON containing validation metrics")
@click.option("-v", "--keep-venv", is_flag=True, required=False, help="Use flag to keep the python virtual environment (python_env in fedn.yaml)")
@click.pass_context
def validate_cmd(ctx, path, input, output, keep_venv):
    """Execute 'validate' entrypoint in fedn.yaml.

    :param ctx:
    :param path: Path to folder containing fedn.yaml
    :type path: str
    """
    path = os.path.abspath(path)
    yaml_file = check_yaml_exists(path)

    config = _read_yaml_file(yaml_file)
    # Check that validate is defined in fedn.yaml under entry_points
    if "validate" not in config["entry_points"]:
        logger.error("No validate command defined in fedn.yaml")
        exit(-1)

    dispatcher = Dispatcher(config, path)
    _ = dispatcher._get_or_create_python_env()
    dispatcher.run_cmd("validate {} {}".format(input, output))
    if not keep_venv:
        delete_virtual_environment(dispatcher)


@run_cmd.command("train")
@click.option("-p", "--path", required=True, help="Path to package directory containing fedn.yaml")
@click.option("-i", "--input", required=True, help="Path to input model parameters")
@click.option("-o", "--output", required=True, help="Path to write the updated model parameters ")
@click.option("-v", "--keep-venv", is_flag=True, required=False, help="Use flag to keep the python virtual environment (python_env in fedn.yaml)")
@click.pass_context
def train_cmd(ctx, path, input, output, keep_venv):
    """Execute 'train' entrypoint in fedn.yaml.

    :param ctx:
    :param path: Path to folder containing fedn.yaml
    :type path: str
    """
    path = os.path.abspath(path)
    yaml_file = check_yaml_exists(path)

    config = _read_yaml_file(yaml_file)
    # Check that train is defined in fedn.yaml under entry_points
    if "train" not in config["entry_points"]:
        logger.error("No train command defined in fedn.yaml")
        exit(-1)

    dispatcher = Dispatcher(config, path)
    _ = dispatcher._get_or_create_python_env()
    dispatcher.run_cmd("train {} {}".format(input, output))
    if not keep_venv:
        delete_virtual_environment(dispatcher)


@run_cmd.command("startup")
@click.option("-p", "--path", required=True, help="Path to package directory containing fedn.yaml")
@click.option("-v", "--keep-venv", is_flag=True, required=False, help="Use flag to keep the python virtual environment (python_env in fedn.yaml)")
@click.pass_context
def startup_cmd(ctx, path, keep_venv):
    """Execute 'startup' entrypoint in fedn.yaml.

    :param ctx:
    :param path: Path to folder containing fedn.yaml
    :type path: str
    """
    path = os.path.abspath(path)
    yaml_file = check_yaml_exists(path)

    config = _read_yaml_file(yaml_file)
    # Check that startup is defined in fedn.yaml under entry_points
    if "startup" not in config["entry_points"]:
        logger.error("No startup command defined in fedn.yaml")
        exit(-1)
    dispatcher = Dispatcher(config, path)
    _ = dispatcher._get_or_create_python_env()
    dispatcher.run_cmd("startup")
    if not keep_venv:
        delete_virtual_environment(dispatcher)


@run_cmd.command("build")
@click.option("-p", "--path", required=True, help="Path to package directory containing fedn.yaml")
@click.option("-v", "--keep-venv", is_flag=True, required=False, help="Use flag to keep the python virtual environment (python_env in fedn.yaml)")
@click.pass_context
def build_cmd(ctx, path, keep_venv):
    """Execute 'build' entrypoint in fedn.yaml.

    :param ctx:
    :param path: Path to folder containing fedn.yaml
    :type path: str
    """
    path = os.path.abspath(path)
    yaml_file = check_yaml_exists(path)

    config = _read_yaml_file(yaml_file)
    # Check that build is defined in fedn.yaml under entry_points
    if "build" not in config["entry_points"]:
        logger.error("No build command defined in fedn.yaml")
        exit(-1)

    dispatcher = Dispatcher(config, path)
    _ = dispatcher._get_or_create_python_env()
    dispatcher.run_cmd("build")
    if not keep_venv:
        delete_virtual_environment(dispatcher)


@run_cmd.command("client")
@click.option("-d", "--discoverhost", required=False, help="Hostname for discovery services(reducer).")
@click.option("-p", "--discoverport", required=False, help="Port for discovery services (reducer).")
@click.option("--token", required=False, help="Set token provided by reducer if enabled")
@click.option("-n", "--name", required=False, default="client" + str(uuid.uuid4())[:8])
@click.option("-i", "--client_id", required=False)
@click.option("--local-package", is_flag=True, help="Enable local compute package")
@click.option("--force-ssl", is_flag=True, help="Force SSL/TLS for REST service")
@click.option("-u", "--dry-run", required=False, default=False)
@click.option("-s", "--secure", required=False, default=False)
@click.option("-pc", "--preshared-cert", required=False, default=False)
@click.option("-v", "--verify", is_flag=True, help="Verify SSL/TLS for REST service")
@click.option("-c", "--preferred-combiner", required=False, type=str, default="", help="url to the combiner or name of the preferred combiner")
@click.option("-va", "--validator", required=False, default=True)
@click.option("-tr", "--trainer", required=False, default=True)
@click.option("-in", "--init", required=False, default=None, help="Set to a filename to (re)init client from file state.")
@click.option("-l", "--logfile", required=False, default=None, help="Set logfile for client log to file.")
@click.option("--heartbeat-interval", required=False, default=2)
@click.option("--reconnect-after-missed-heartbeat", required=False, default=30)
@click.option("--verbosity", required=False, default="INFO", type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], case_sensitive=False))
@click.pass_context
def client_cmd(
    ctx,
    discoverhost,
    discoverport,
    token,
    name,
    client_id,
    local_package,
    force_ssl,
    dry_run,
    secure,
    preshared_cert,
    verify,
    preferred_combiner,
    validator,
    trainer,
    init,
    logfile,
    heartbeat_interval,
    reconnect_after_missed_heartbeat,
    verbosity,
):
    """:param ctx:
    :param discoverhost:
    :param discoverport:
    :param token:
    :param name:
    :param client_id:
    :param remote:
    :param dry_run:
    :param secure:
    :param preshared_cert:
    :param verify_cert:
    :param preferred_combiner:
    :param init:
    :param logfile:
    :param hearbeat_interval
    :param reconnect_after_missed_heartbeat
    :param verbosity
    :return:
    """
    remote = False if local_package else True
    config = {
        "discover_host": discoverhost,
        "discover_port": discoverport,
        "token": token,
        "name": name,
        "client_id": client_id,
        "remote_compute_context": remote,
        "force_ssl": force_ssl,
        "dry_run": dry_run,
        "secure": secure,
        "preshared_cert": preshared_cert,
        "verify": verify,
        "preferred_combiner": preferred_combiner,
        "validator": validator,
        "trainer": trainer,
        "logfile": logfile,
        "heartbeat_interval": heartbeat_interval,
        "reconnect_after_missed_heartbeat": reconnect_after_missed_heartbeat,
        "verbosity": verbosity,
    }

    click.echo(
        click.style("\n*** fedn run client is deprecated and will be removed. Please use fedn client start instead. ***\n", blink=True, bold=True, fg="red")
    )

    if init:
        apply_config(init, config)
        click.echo(f"\nClient configuration loaded from file: {init}")
        click.echo("Values set in file override defaults and command line arguments...\n")
    try:
        validate_client_config(config)
    except InvalidClientConfig as e:
        click.echo(f"Error: {e}")
        return

    client = Client(config)
    client.run()


@run_cmd.command("combiner")
@click.option("-d", "--discoverhost", required=False, help="Hostname for discovery services (reducer).")
@click.option("-p", "--discoverport", required=False, help="Port for discovery services (reducer).")
@click.option("-t", "--token", required=False, help="Set token provided by reducer if enabled")
@click.option("-n", "--name", required=False, default="combiner" + str(uuid.uuid4())[:8], help="Set name for combiner.")
@click.option("-h", "--host", required=False, default="combiner", help="Set hostname.")
@click.option("-i", "--port", required=False, default=12080, help="Set port.")
@click.option("-f", "--fqdn", required=False, default=None, help="Set fully qualified domain name")
@click.option("-s", "--secure", is_flag=True, help="Enable SSL/TLS encrypted gRPC channels.")
@click.option("-v", "--verify", is_flag=True, help="Verify SSL/TLS for REST discovery service (reducer)")
@click.option("-c", "--max_clients", required=False, default=30, help="The maximal number of client connections allowed.")
@click.option("-in", "--init", required=False, default=None, help="Path to configuration file to (re)init combiner.")
@click.pass_context
def combiner_cmd(ctx, discoverhost, discoverport, token, name, host, port, fqdn, secure, verify, max_clients, init):
    """:param ctx:
    :param discoverhost:
    :param discoverport:
    :param token:
    :param name:
    :param hostname:
    :param port:
    :param secure:
    :param max_clients:
    :param init:
    """
    config = {
        "discover_host": discoverhost,
        "discover_port": discoverport,
        "token": token,
        "host": host,
        "port": port,
        "fqdn": fqdn,
        "name": name,
        "secure": secure,
        "verify": verify,
        "max_clients": max_clients,
    }

    click.echo(
        click.style("\n*** fedn run combiner is deprecated and will be removed. Please use fedn combiner start instead. ***\n", blink=True, bold=True, fg="red")
    )

    if init:
        apply_config(init, config)
        click.echo(f"\nCombiner configuration loaded from file: {init}")
        click.echo("Values set in file override defaults and command line arguments...\n")

    from fedn.network.combiner.combiner import Combiner

    combiner = Combiner(config)
    combiner.run()
