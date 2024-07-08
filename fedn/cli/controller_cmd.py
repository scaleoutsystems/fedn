import click
from .main import main
from fedn.network.api.server import app
from fedn.common.config import get_controller_config

@main.group("controller")
@click.pass_context
def controller_cmd(ctx):
    """:param ctx:
    """
    pass
@controller_cmd.command("start")
@click.option("-H", "--host", required=False, default="0.0.0.0", help="Host address of controller")
@click.pass_context
def controller_cmd(ctx,host):
    config = get_controller_config()
    port = config["port"]
    debug = config["debug"]
    app.run(debug=debug, port=port,host=host)
