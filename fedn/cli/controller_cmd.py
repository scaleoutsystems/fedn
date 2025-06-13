import click

from fedn.cli.main import main
from fedn.cli.shared import apply_config
from fedn.common.config import get_modelstorage_config, get_network_config, get_statestore_config
from fedn.network.controller.control import Control
from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.s3.repository import Repository


@main.group("controller")
@click.pass_context
def controller_cmd(ctx):
    """:param ctx:"""
    pass


@controller_cmd.command("start")
@click.option("-h", "--host", required=False, default="controller", help="Set hostname.")
@click.option("-i", "--port", required=False, default=12090, help="Set port.")
@click.option("-s", "--secure", is_flag=True, help="Enable SSL/TLS encrypted gRPC channels.")
@click.option("-in", "--init", required=False, default=None, help="Path to configuration file.")
@click.pass_context
def controller_cmd(ctx, host, port, secure, init):
    config = {
        "host": host,
        "port": port,
        "secure": secure,
    }

    if init:
        apply_config(init, config)
        click.echo(f"\nController configuration loaded from file: {init}")

    network_id = get_network_config()
    modelstorage_config = get_modelstorage_config()
    statestore_config = get_statestore_config()

    db = DatabaseConnection(statestore_config, network_id)
    repository = Repository(modelstorage_config["storage_config"], storage_type=modelstorage_config["storage_type"])
    controller = Control(config, network_id, repository, db)
    controller.run()
