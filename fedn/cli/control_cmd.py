import click

from .main import main


@main.group('control')
@click.pass_context
def control_cmd(ctx):
    # if daemon:
    #    print('{} NYI should run as daemon...'.format(__file__))
    pass


@control_cmd.command('package')
@click.option('-r', '--reducer', required=False)
@click.option('-p', '--port', required=False)
@click.option('-t', '--token', required=False)
@click.option('-n', '--name', required=False, default=None)
@click.option('-u', '--upload', required=False, default=None)
@click.option('-v', '--validate', required=False, default=False)
@click.option('-d', '--cwd', required=False, default=None)
@click.pass_context
def package_cmd(ctx, reducer, port, token, name, upload, validate, cwd):
    if not cwd:
        import os
        cwd = os.getcwd()

    print("CONTROL: Bundling {} dir for distribution. Please wait for operation to complete..".format(cwd))

    if not name:
        from datetime import datetime
        name = str(os.path.basename(cwd)) + '-' + datetime.today().strftime('%Y-%m-%d-%H%M%S')
        
    config = {'host': reducer, 'port': port, 'token': token, 'name': name,
              'cwd': cwd}

    from fedn.common.control.package import Package
    package = Package(config)

    print("CONTROL: Bundling package..")
    package.package(validate=validate)
    print("CONTROL: Bundle completed\nCONTROL: Resulted in: {}.tar.gz".format(name))
    if upload:
        print("CONTROL: started upload")
        package.upload()
        print("CONTROL: upload finished!")
    else:
        print("CONTROL: set --upload flag along with --reducer and --port if you want to upload directly from client.")


@control_cmd.command('unpack')
@click.option('-r', '--reducer', required=True)
@click.option('-p', '--port', required=True)
@click.option('-t', '--token', required=True)
@click.option('-n', '--name', required=False, default=None)
@click.option('-d', '--download', required=False, default=None)
@click.option('-v', '--validate', required=False, default=False)
@click.option('-c', '--cwd', required=False, default=None)
@click.pass_context
def unpack_cmd(ctx, reducer, port, token, name, download, validate, cwd):
    import os
    if not cwd:
        cwd = os.getcwd()

    config = {'host': reducer, 'port': port, 'token': token, 'name': name,
              'cwd': cwd}

    from fedn.common.control.package import PackageRuntime
    package = PackageRuntime(cwd, os.path.join(cwd, 'client'))
    package.download(reducer, port, token)
    package.unpack()


@control_cmd.command('template')
@click.pass_context
def template_cmd(ctx):
    print("TODO: generate template")
    pass


@control_cmd.command('start')
@click.option('-r', '--reducer', required=True)
@click.option('-p', '--port', required=True)
@click.option('-t', '--token', required=True)
@click.pass_context
def control_cmd(ctx, reducer, port, token):
    pass

