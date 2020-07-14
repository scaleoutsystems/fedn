import click
import os
from .main import main
from prettytable import PrettyTable

from scaleout.project import init_project as init_p
from scaleout.studioclient import StudioClient

def init_project(project_dir):
    init_p(project_dir)

@click.option('--daemon',
              is_flag=True,
              help=(
                      "Specify to run in daemon mode."
              )
              )

@main.group('project')
@click.pass_context
def project_cmd(ctx, daemon):
  if daemon:
      print('{} NYI should run as daemon...'.format(__file__))

@project_cmd.command('init')
@click.pass_context
def init_cmd(ctx):
	""" Create project template files in the current working directory. """
	project_dir = ctx.obj['PROJECT_DIR']
	init_project(project_dir)

@project_cmd.command('fetch')
@click.pass_context
def fetch_cmd(ctx):
	""" Stage project components locally.  """


@project_cmd.command('list')
@click.pass_context
def list_cmd(ctx):
	""" List projects. """
	client = ctx['CONTROLLER']


@project_cmd.group('create')
@click.pass_context
def project_create_cmd(ctx):
    pass

@project_create_cmd.command('deploymentdefinition')
@click.option('-n', '--name', required=True)
@click.option('-d', '--definition', required=True)
@click.option('-b', '--bucket', required=True)
@click.option('-f', '--filepath', required=True)
@click.option('-p', '--path_predict')
@click.pass_context
def project_create_deployment_definition(ctx, name, definition, bucket, filepath, path_predict=''):
    client = ctx.obj['CLIENT']
    client.create_deployment_definition(name, definition, bucket, filepath, path_predict)

@project_cmd.group('list')
@click.pass_context
def project_list_cmd(ctx):
    pass

@project_list_cmd.command('deploymentdefinition')
@click.pass_context
def project_list_deployment_definition(ctx):
    client = ctx.obj['CLIENT']
    deploymentdefinitions = client.list_deployment_definitions()
    x = PrettyTable()
    x.field_names = ["Name","Bucket","File"]
    for d in deploymentdefinitions:
        x.add_row([d["name"],d["bucket"],d["filename"]])
    print(x)


@project_cmd.group('get')
@click.pass_context
def project_get_cmd(ctx):
    pass

@project_get_cmd.command('deploymentdefinition')
@click.option('-n', '--name', required=True)
@click.pass_context
def project_get_deployment_definition(ctx, name):
    client = ctx.obj['CLIENT']
    dd = client.get_deployment_definition(name)
    x = PrettyTable()
    x.field_names = ["Name","Definition","Bucket","File"]
    x.add_row([dd[0]["name"],dd[0]["definition"],dd[0]["bucket"],dd[0]["filename"]])
    print(x)