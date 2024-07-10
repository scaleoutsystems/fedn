import importlib.metadata

import click

CONTEXT_SETTINGS = dict(
    # Support -h as a shortcut for --help
    help_option_names=["-h", "--help"],
)

# Dynamically get the version of the package
try:
    version = importlib.metadata.version("fedn")
except importlib.metadata.PackageNotFoundError:
    version = "unknown"


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version)
@click.pass_context
def main(ctx):
    """:param ctx:"""
    ctx.obj = dict()
