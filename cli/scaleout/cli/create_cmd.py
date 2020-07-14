import click
from .main import main
import requests
from prettytable import PrettyTable
from scaleout.studioclient import StudioClient 


@click.option('--daemon',
              is_flag=True,
              help=(
                      "Specify to run in daemon mode."
              )
              )

@main.group('create')
@click.pass_context
def create_cmd(ctx, daemon):
  if daemon:
      print('{} NYI should run as daemon...'.format(__file__))

@create_cmd.command('model')
@click.option('-m', '--model', required=True)
@click.option('-n', '--name', required=True)
@click.option('-t', '--tag', required=False,default="latest")
@click.option('-d', '--description', required=False,default="")
@click.pass_context
def create_model_cmd(ctx, model, name, tag, description):
  """ Publish a model. """
  client = ctx.obj['CLIENT']
  client.create_model(model, name, tag, description)

@create_cmd.command('deploymentdefinition')
@click.option('-n', '--name', required=True)
@click.option('-f', '--filepath', required=True)
@click.option('-p', '--path_predict')
@click.pass_context
def create_deployment_definition(ctx, name, filepath, path_predict=''):
    """ Create a depliyment definition. """
    client = ctx.obj['CLIENT']
    client.create_deployment_definition(name, filepath, path_predict)

@create_cmd.command('deployment')
@click.option('-m', '--model', required=True)
@click.option('-t', '--model-tag', default='latest')
@click.option('-d', '--deploymentdefinition', required=True)
@click.pass_context
def create_deployment_cmd(ctx, model, deploymentdefinition, model_tag='latest'):
    client = ctx.obj['CLIENT']
    client.deploy_model(model, model_tag, deploymentdefinition)


# Create project

# Create dataset