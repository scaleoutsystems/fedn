import os
import sys

import click

from .main import main

envs = [
    "FEDN_PROTOCOL",
    "FEDN_HOST",
    "FEDN_PORT",
    "FEDN_TOKEN",
    "FEDN_AUTH_SCHEME"
]


@main.group('config', invoke_without_command=True)
@click.pass_context
def config_cmd(ctx):
    """
    :param ctx:
    """
    if ctx.invoked_subcommand is None:
        click.echo('\n--- FEDn Cli Configuration ---\n')
        click.echo('Current configuration:\n')

        for env in envs:
            value = os.environ.get(env)
            click.echo(f'{env}: {value or "Not set"}')
        click.echo('\n')


@config_cmd.command('generate')
@click.option('--protocol', prompt="Protocol", default=lambda: os.environ.get("FEDN_PROTOCOL", ""))
@click.option('--host', prompt="Host", default=lambda: os.environ.get("FEDN_HOST", ""))
@click.option('--port', prompt="Port", default=lambda: os.environ.get("FEDN_PORT", ""))
@click.option('--token', prompt="Token", default=lambda: os.environ.get("FEDN_TOKEN", ""))
@click.option('--scheme', prompt="Scheme", default=lambda: os.environ.get("FEDN_AUTH_SCHEME", ""))
def set_config(protocol: str, host: str, port: int, token: str, scheme: str):
    """
    - Get configuration script from user input that can be used to set environment variables.
    return: None
    """
    click.echo('\n--- FEDn Cli Configuration ---\n')
    click.echo("Paste the following and run in your terminal\n")
    click.echo("-------------------\n")

    os_name = sys.platform

    prefix = "export" if os_name != 'win32' else "set"
    values = [protocol, host, port, token, scheme]

    for i in range(len(envs)):
        env = envs[i]
        current_value = os.environ.get(env)
        new_value = values[i]

        if new_value and new_value != current_value:
            click.echo(f"{prefix} {envs[i]}={values[i]};")


@config_cmd.command('clear')
def clear_config():
    """
    - Get configuration script from user input that can be used to remove environment variables.
    return: None
    """
    click.echo('\n--- FEDn Cli Configuration ---\n')
    click.echo("Paste the following and run in your terminal\n")
    click.echo("-------------------\n")

    os_name = sys.platform

    if os_name == 'win32':

        for env in envs:
            click.echo(f"set {env}=;")
    else:
        for env in envs:
            click.echo(f"unset {env};")
