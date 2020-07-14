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

@main.group('model')
@click.pass_context
def model_cmd(ctx, daemon):
  if daemon:
      print('{} NYI should run as daemon...'.format(__file__))



@model_cmd.command('show')
@click.option('-m', '--model', required=True)
@click.pass_context
def show_cmd(ctx,model):
  """ Show model details. """
  client = ctx.obj['CLIENT']
  model = client.show_model(model)
  x = PrettyTable()
  #x.field_names = list(model.keys())
  x.add_column("Property",list(model.keys()))
  x.add_column("Value",list(model.values()))
  print(x)


@model_cmd.command('predict')
@click.option('-i', '--input_file', required=True)
@click.option('-n', '--name', required=True)
@click.option('-v', '--version', required=True)
@click.pass_context
def predict_cmd(ctx, input_file, name, version):
    client = ctx.obj['CLIENT']
    client.predict(input_file, name, version)


# Create group
@model_cmd.group('create')
@click.pass_context
def model_create_cmd(ctx):
    pass

@model_create_cmd.command('deployment')
@click.option('-m', '--model', required=True)
@click.option('-n', '--name', required=True)
@click.option('-c', '--context', required=True)
@click.option('-v', '--version', required=True)
@click.pass_context
def model_cmd_deploy(ctx, model, context, name, version):
    client = ctx.obj['CLIENT']
    client.deploy_model(model, context, name, version)


# List group

@model_cmd.group('list')
@click.pass_context
def model_list_cmd(ctx):
    pass

@model_list_cmd.command('deployments')
@click.pass_context
def deploy_list_cmd(ctx):
    client = ctx.obj['CLIENT']
    deployments = client.list_deployments()
    x = PrettyTable()
    x.field_names = ["Name","Image","InvocationCount"]
    for d in deployments:
        x.add_row([d["name"],d["image"],d["invocationCount"]])
    print(x)

@model_list_cmd.command('all')
@click.pass_context
def list_cmd(ctx):
    """ List all models and show their status and endpoints """
    client = ctx.obj['CLIENT']
    models = client.list_models()

    x = PrettyTable()
    x.field_names = ["Name","Tag","Created"]
    for m in models:
        x.add_row([m["name"],m["tag"],m["uploaded_at"]])
    print(x)

###########################

# @model_cmd.command('predict')
# @click.option('-d', '--deployment', required=True)
# @click.option('-i', '--input', required=True)
# @click.pass_context
# def cmd_predict(ctx):


@click.option('-m','--model_id',required=True)
@click.option('-t','--tag')
@click.option('-o','--output',default="model.out")
@model_cmd.command('download')
@click.pass_context
def download_cmd(ctx, model_id, tag, output):
    """ Download a model. """
    # TODO: Use model tag, default to latest 
    client = ctx.obj['CLIENT']
    repository = client.get_repository()
    repository.bucket = 'models'
    obj = repository.get_artifact(model_id)

    with open(output, 'wb') as fh:
      fh.write(obj)

@click.option('-n', '--name', required=True)
@click.option('-t', '--tag', required=False)
@model_cmd.command('delete')
@click.pass_context
def delete_cmd(ctx, name,tag):
    """ Delete a model. """
    # TODO: It should probably not be possible to delete ANY model (e.g. seed models)
    client = ctx.obj['CLIENT']
    client.delete_model(name,tag)
    #repository = client.get_repository()
    #repository.bucket = 'models'
    #repository.delete_artifact(instance_name)
