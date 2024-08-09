import os

import click

from .main import main

envs = [
    {"name": "FEDN_CONTROLLER_PROTOCOL", "description": "The protocol to use for communication with the controller."},
    {"name": "FEDN_CONTROLLER_HOST", "description": "The host to use for communication with the controller."},
    {"name": "FEDN_CONTROLLER_PORT", "description": "The port to use for communication with the controller."},
    {"name": "FEDN_AUTH_TOKEN", "description": "The authentication token to use for communication with the controller and combiner."},
    {"name": "FEDN_AUTH_SCHEME", "description": "The authentication scheme to use for communication with the controller and combiner."},
    {
        "name": "FEDN_CONTROLLER_URL",
        "description": "The URL of the controller. Overrides FEDN_CONTROLLER_PROTOCOL, FEDN_CONTROLLER_HOST and FEDN_CONTROLLER_PORT.",
    },
    {"name": "FEDN_PACKAGE_EXTRACT_DIR", "description": "The directory to extract packages to."},
]


@main.group("config", invoke_without_command=True)
@click.pass_context
def config_cmd(ctx):
    """- Configuration commands for the FEDn CLI.
    """
    if ctx.invoked_subcommand is None:
        click.echo("\n--- FEDn Cli Configuration ---\n")
        click.echo("Current configuration:\n")

        for env in envs:
            name = env["name"]
            value = os.environ.get(name)
            click.echo(f'{name}: {value or "Not set"}')
            click.echo(f'{env["description"]}\n')
        click.echo("\n")
