from fedn.utils.dist import get_version
import click

CONTEXT_SETTINGS = dict(
    # Support -h as a shortcut for --help
    help_option_names=["-h", "--help"],
)

version=get_version("fedn")


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version)
@click.pass_context
def main(ctx):
    """:param ctx:"""
    ctx.obj = dict()
