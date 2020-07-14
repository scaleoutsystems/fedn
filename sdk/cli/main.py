import click

import logging

logging.basicConfig(format='%(asctime)s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')  # , level=logging.DEBUG)

CONTEXT_SETTINGS = dict(
    # Support -h as a shortcut for --help
    help_option_names=['-h', '--help'],
)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    '--project',
    '-p',
    'project_dir',
    help=(
            "Supply a project directory"
    ),
    type=click.Path(exists=False, dir_okay=True),
)
@click.option(
    '--login/--no-login',
    default=True
)
@click.option(
    '--endpoints/--no-endpoints',
    default=True
)
@click.pass_context
def main(ctx, project_dir, login, endpoints):
    ctx.obj = dict()
    ctx.obj['PROJECT_DIR'] = project_dir

    if ctx.invoked_subcommand not in ('init',):
        # TODO add support for cwd change, config-file specification
        #        from scaleout.project import Project
        from scaleout.errors import InvalidConfigurationError
        try:
            from scaleout.project import Project
            ctx.obj['PROJECT'] = Project(project_dir)
        except InvalidConfigurationError as e:
            print("InvalidConfigurationError: {}".format(e), flush=True)
        # ctx.obj['CONTROLLER'] = controller
        # ctx.obj['CLIENT'] = StudioClient(login=login, endpoints=endpoints)
