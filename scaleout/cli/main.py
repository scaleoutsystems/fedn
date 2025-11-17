from scaleoututil.utils.dist import get_version
import click

from scaleoututil.utils.url import parse_url, build_url


from scaleout.cli.shared import set_context

CONTEXT_SETTINGS = dict(
    # Support -h as a shortcut for --help
    help_option_names=["-h", "--help"],
)

version = get_version()


@click.group(
    context_settings=CONTEXT_SETTINGS,
    help="Scaleout command-line interface for starting clients and interacting with API resources such as models, sessions and more.",
    epilog="Use 'scaleout COMMAND --help' for detailed help on a command.",
    no_args_is_help=True,
)
@click.version_option(version)
@click.pass_context
def main(ctx):
    """:param ctx:"""
    ctx.obj = dict()


@main.command("login")
@click.option("-u", "--username", required=False, default=None, help="scaleout username")
@click.option("-P", "--password", required=False, default=None, help="scaleout password")
@click.option("-p", "--protocol", required=False, default=None, help="Communication protocol of API (http or https)")
@click.option("-t", "--token", required=False, default=None, help="Authentication token")
@click.option("-H", "--host", required=True, help="Hostname of the Scaleout API (e.g. api.scaleout.scaleoutsystems.com). May include protocol and port.")
@click.option("-P", "--port", required=False, default=None, type=int, help="Port of the Scaleout API.")
@click.pass_context
def login_cmd(ctx, protocol: str, host: str, port: int, token: str, username: str, password: str):
    """Login to Scaleout (set context)."""
    # Step 1: Display welcome message
    click.secho("Welcome to Scaleout!", fg="green")
    _protocol, _host, _port, _endpoint = parse_url(host)
    protocol = protocol or _protocol
    host = _host
    port = port or _port
    endpoint = _endpoint

    if port:
        if port < 1 or port > 65535:
            raise ValueError("Port number must be between 1 and 65535.")

    if protocol:
        if protocol not in ["http", "https"]:
            click.secho("Unsupported protocol. Use 'http' or 'https'.", fg="red")
            return
    if endpoint:
        click.secho("Host should not include an endpoint/path.", fg="red")
        return

    host = build_url(protocol, host, port, "")

    # TODO: Verify connection to FEDn API
    set_context(host=host, token=token)
    click.secho(f"Successfully set context to host: {host}", fg="green")
