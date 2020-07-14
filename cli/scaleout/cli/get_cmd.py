import click
from .main import main
import requests
from scaleout.studioclient import StudioClient 
from .helpers import create_table

@click.option('--daemon',
              is_flag=True,
              help=(
                      "Specify to run in daemon mode."
              )
              )

@main.group('get')
@click.pass_context
def get_cmd(ctx, daemon):
    if daemon:
        print('{} NYI should run as daemon...'.format(__file__))

@get_cmd.command('models')
@click.pass_context
def get_models_cmd(ctx):
    """ List all models and show their status and endpoints """
    names = ["Name","Tag","Created"]
    keys = ['name', 'tag', 'uploaded_at']
    create_table(ctx, 'models', names, keys)

@get_cmd.command('deployments')
@click.pass_context
def get_deployments_cmd(ctx):
    # client = ctx.obj['CLIENT']
    # client.list_deployments()
    names = ["Name","Tag", "Endpoint"]
    keys = ["name", "tag", "endpoint"]
    create_table(ctx, 'deploymentInstances', names, keys)

@get_cmd.command('deploymentdefinitions')
@click.pass_context
def get_deploymentdefinitions_cmd(ctx):
    names = ["Name"]
    keys = ["name"]
    create_table(ctx, "deploymentDefinitions", names, keys)

@get_cmd.command('projects')
@click.pass_context
def get_projects_cmd(ctx):
    names = ["Name","Created", "Last updated"]
    keys = ["name", "created_at", "updated_at"]
    create_table(ctx, "projects", names, keys)


# alliance

# dataset