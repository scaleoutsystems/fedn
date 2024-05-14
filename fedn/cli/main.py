import click

CONTEXT_SETTINGS = dict(
    # Support -h as a shortcut for --help
    help_option_names=["-h", "--help"],
)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.pass_context
def main(ctx):
    """:param ctx:
    """
    ctx.obj = dict()
