import os

import click
import requests
import yaml

from .main import main

home_dir = os.path.expanduser("~")


@main.group("project")
@click.pass_context
def project_cmd(ctx):
    """:param ctx:"""
    pass


@click.option("-p", "--protocol", required=False, default="https", help="Communication protocol")
@click.option("-H", "--host", required=False, default="fedn.scaleoutsystems.com", help="Hostname of controller (api)")
@click.option("-t", "--token", required=False, help="User access token")
@project_cmd.command("create")
@click.pass_context
def create_project(ctx, protocol: str = None, host: str = None, token: str = None):
    """Create project.
    :param ctx:
    """
    url = f"{protocol}://{host}/api/v1/projects/create"
    headers = {}

    if token:
        headers = {"Content-Type": "application/x-www-form-urlencoded", "Authorization": f"Bearer {token}"}

    name = input("Please enter a project name: ")
    description = input("Please enter a project description (optional): ")

    # Call the authentication API
    try:
        requests.post(url, data={"name": name, "description": description}, headers=headers)
    except requests.exceptions.RequestException as e:
        click.secho(str(e), fg="red")

    click.secho("Project created.", fg="green")


@click.option("-s", "--slug", required=True, help="Slug name of project.")
@click.option("-p", "--protocol", required=False, default="https", help="Communication protocol")
@click.option("-H", "--host", required=False, default="fedn.scaleoutsystems.com", help="Hostname of controller (api)")
@click.option("-t", "--token", required=False, help="User access token")
@project_cmd.command("delete")
@click.pass_context
def delete_project(ctx, slug: str = None, protocol: str = None, host: str = None, token: str = None):
    """Create project.

    :param ctx:
    :param name:
    :param description:
    """
    url = f"{protocol}://{host}/api/internal/app/projects/delete/{slug}"
    headers = {}

    if token:
        headers = {"Authorization": f"Bearer {token}"}

    # Call the authentication API
    try:
        requests.delete(url, headers=headers)
    except requests.exceptions.RequestException as e:
        click.secho(str(e), fg="red")

    # if deleted project is active, activate another
    set_active_project(None, protocol, host)
    click.secho("Project successfully deleted.", fg="green")


@click.option("-p", "--protocol", required=False, default="https", help="Communication protocol")
@click.option("-H", "--host", required=False, default="fedn.scaleoutsystems.com", help="Hostname of controller (api)")
@click.option("-t", "--token", required=False, help="User access token")
@project_cmd.command("list")
@click.pass_context
def list_projects(ctx, protocol: str = None, host: str = None, token: str = None):
    """Return:
    ------
    - result: list of packages

    """
    url = f"{protocol}://{host}/api/v1/projects"
    headers = {}

    if token:
        headers = {"Authorization": f"Bearer {token}"}

    context_path = os.path.join(home_dir, ".fedn")
    try:
        with open(f"{context_path}/context.yaml", "r") as yaml_file:
            context_data = yaml.safe_load(yaml_file)
    except Exception as e:
        print(f"Error: Failed to read YAML file. Details: {e}")

    active_project = context_data.get("Active project slug")

    try:
        response = requests.get(url, headers=headers)
        response_json = response.json()
        for i in response_json:
            project_name = i.get("slug")
            if project_name == active_project:
                click.secho(project_name, fg="green")
            else:
                click.secho(project_name)
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url}")


@click.option("-s", "--slug", required=True, help="Slug name of project.")
@click.option("-p", "--protocol", required=False, default="https", help="Communication protocol")
@click.option("-H", "--host", required=False, default="fedn.scaleoutsystems.com", help="Hostname of controller (api)")
@click.option("-t", "--token", required=False, help="User access token")
@project_cmd.command("get")
@click.pass_context
def get_project(ctx, slug: str = None, protocol: str = None, host: str = None, token: str = None):
    """Return:
    ------
    - result: project with given slug

    """
    url = f"{protocol}://{host}/api/v1/projects"
    headers = {}

    if token:
        headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers)
        response_json = response.json()
        if len(response_json) == 1:
            print(response_json[0].get("slug"))
        else:
            project_found = False
            for i in response_json:
                i_name = i.get("slug")
                if i_name.lower() == slug.lower():
                    project_found = True
                    print(i)
            if not project_found:
                click.secho(f"No project with name {slug} exists for this account.", fg="red")
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url}")


@click.option("-s", "--slug", required=True, help="Slug name of project.")
@click.option("-p", "--protocol", required=False, default="https", help="Communication protocol")
@click.option("-H", "--host", required=False, default="fedn.scaleoutsystems.com", help="Hostname of controller (api)")
@project_cmd.command("activate")
@click.pass_context
def set_active_project(ctx, slug: str = None, protocol: str = None, host: str = None):
    """Set active project.

    :param ctx:
    :param slug:
    """
    url_projects = f"{protocol}://{host}/api/v1/projects"
    url_project_token = f"{protocol}://{host}/api/v1/admin-token"
    context_path = os.path.join(home_dir, ".fedn")
    try:
        with open(f"{context_path}/context.yaml", "r") as yaml_file:
            context_data = yaml.safe_load(yaml_file)
    except Exception as e:
        print(f"Error: Failed to read YAML file. Details: {e}")

    user_access_token = context_data.get("User tokens").get("access")

    headers_projects = {}

    if user_access_token:
        headers_projects = {"Authorization": f"Bearer {user_access_token}"}

    try:
        response_projects = requests.get(url_projects, headers=headers_projects)
        projects_response_json = response_projects.json()
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url_projects}")

    for i in projects_response_json:
        if i.get("slug") == slug:
            headers_projects["X-Project-Slug"] = i.get("slug")

    try:
        response_project_tokens = requests.get(url_project_token, headers=headers_projects)
        project_tokens = response_project_tokens.json()
        context_data["Active project tokens"] = project_tokens
        context_data["Active project slug"] = slug
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url_project_token}")

    try:
        with open(f"{context_path}/context.yaml", "w") as yaml_file:
            yaml.dump(context_data, yaml_file, default_flow_style=False)  # Add access and refresh tokens to context yaml file
    except Exception as e:
        print(f"Error: Failed to write to YAML file. Details: {e}")

    click.secho(f"Project with slug {slug} was succsessfully activated.", fg="green")
