import os

import click

from .main import main


@main.group('config', invoke_without_command=True)
@click.pass_context
def config_cmd(ctx):
    """
    :param ctx:
    """
    if ctx.invoked_subcommand is None:
        _protocol = os.environ.get('FEDN_PROTOCOL')
        _host = os.environ.get('FEDN_HOST')
        _port = os.environ.get('FEDN_PORT')
        _token = os.environ.get("FEDN_TOKEN")
        _scheme = os.environ.get("FEDN_AUTH_SCHEME")

        click.echo('\n--- FEDn Cli Configuration ---\n')
        click.echo('Current configuration:\n')

        click.echo(f'FEDN_PROTOCOL: {_protocol or "Not set"}')
        click.echo(f'FEDN_HOST: {_host or "Not set"}')
        click.echo(f'FEDN_PORT: {_port or "Not set"}')
        click.echo(f'FEDN_TOKEN: {_token or "Not set"}')
        click.echo(f'FEDN_AUTH_SCHEME: {_scheme or "Not set"}\n')


@config_cmd.command('set')
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
    click.echo("\n-------------------\n")

    if protocol:
        click.echo(f"\nexport FEDN_PROTOCOL={protocol}")

    if host:
        click.echo(f"export FEDN_HOST={host}")

    if port:
        click.echo(f"export FEDN_PORT={port}")

    if token:
        click.echo(f"export FEDN_TOKEN={token}")

    if scheme:
        click.echo(f"export FEDN_AUTH_SCHEME={scheme}")


@config_cmd.command('clear')
def clear_config():
    """
    - Get configuration script from user input that can be used to remove environment variables.
    return: None
    """
    click.echo("\n-------------------\n")
    click.echo("unset FEDN_PROTOCOL")
    click.echo("unset FEDN_HOST")
    click.echo("unset FEDN_PORT")
    click.echo("unset FEDN_TOKEN")
    click.echo("unset FEDN_AUTH_SCHEME")
