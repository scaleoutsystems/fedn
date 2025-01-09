import os

import click
import requests
import yaml

from .main import main
from .shared import STUDIO_DEFAULTS, get_api_url, get_token, print_response

home_dir = os.path.expanduser("~")


@main.group("project")
@click.pass_context
def project_cmd(ctx):
    """:param ctx:"""
    pass


@click.option("-s", "--slug", required=True, help="Slug name of project.")
@click.option("-p", "--protocol", required=False, default=STUDIO_DEFAULTS["protocol"], help="Communication protocol of studio (api)")
@click.option("-H", "--host", required=False, default=STUDIO_DEFAULTS["host"], help="Hostname of studio (api)")
@click.option("-t", "--token", required=False, help="User access token")
@project_cmd.command("delete")
@click.pass_context
def delete_project(ctx, slug: str = None, protocol: str = None, host: str = None, token: str = None):
    """Delete project."""
    # Check if project with given slug exists
    studio_api = True
    url = get_api_url(protocol=protocol, host=host, port=None, endpoint=f"projects/{slug}", usr_api=studio_api)
    headers = {}

    _token = get_token(token, False)

    if _token:
        headers["Authorization"] = _token
    try:
        response = requests.get(url, headers=headers)
        response_json = response.json()
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url}")
    if response_json.get("error") is None:
        # Check if user wants to delete project with given slug
        user_input = input(f"Are you sure you want to delete project with slug {slug} (y/n)?: ")
        if user_input == "y":
            url = get_api_url(protocol=protocol, host=host, port=None, endpoint=f"projects/delete/{slug}", usr_api=studio_api)
            headers = {}

            _token = get_token(token, True)

            if _token:
                headers["Authorization"] = _token
            # Call the authentication API
            try:
                requests.delete(url, headers=headers)
                click.secho(f"Project with slug {slug} has been removed.", fg="green")
            except requests.exceptions.RequestException as e:
                click.echo(str(e), fg="red")
            activate_project(None, protocol, host)
    else:
        click.secho(f"No project with slug '{slug}' exists.", fg="red")


@click.option("-p", "--protocol", required=False, default=STUDIO_DEFAULTS["protocol"], help="Communication protocol of studio (api)")
@click.option("-H", "--host", required=False, default=STUDIO_DEFAULTS["host"], help="Hostname of studio (api)")
@click.option("-t", "--token", required=False, help="User access token")
@project_cmd.command("create")
@click.pass_context
def create_project(ctx, protocol: str = None, host: str = None, token: str = None):
    """Create project.
    :param ctx:
    """
    # Check if user can create project
    studio_api = True
    url = get_api_url(protocol=protocol, host=host, port=None, endpoint="projects/create", usr_api=studio_api)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    _token = get_token(token, True)

    if _token:
        headers["Authorization"] = _token

    name = input("Please enter a project name: ")
    description = input("Please enter a project description (optional): ")
    if len(name) > 46 or len(description) >= 255:
        click.secho("Project name or description too long.", fg="red")
    else:
        # Call the authentication API
        try:
            requests.post(url, data={"name": name, "description": description}, headers=headers)
        except requests.exceptions.RequestException as e:
            click.secho(str(e), fg="red")
        click.secho("Project created.", fg="green")


@click.option("-p", "--protocol", required=False, default=STUDIO_DEFAULTS["protocol"], help="Communication protocol of studio (api)")
@click.option("-H", "--host", required=False, default=STUDIO_DEFAULTS["host"], help="Hostname of studio (api)")
@click.option("-t", "--token", required=False, help="User access token")
@project_cmd.command("list")
@click.pass_context
def list_projects(ctx, protocol: str = None, host: str = None, token: str = None):
    """Return:
    ------
    - result: list of packages

    """
    studio_api = True
    url = get_api_url(protocol=protocol, host=host, port=None, endpoint="projects", usr_api=studio_api)
    headers = {}

    _token = get_token(token, True)

    if _token:
        headers["Authorization"] = _token
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
@click.option("-p", "--protocol", required=False, default=STUDIO_DEFAULTS["protocol"], help="Communication protocol of studio (api)")
@click.option("-H", "--host", required=False, default=STUDIO_DEFAULTS["host"], help="Hostname of studio (api)")
@click.option("-t", "--token", required=False, help="User access token")
@project_cmd.command("get")
@click.pass_context
def get_project(ctx, slug: str = None, protocol: str = None, host: str = None, token: str = None):
    """Return:
    ------
    - result: project with given slug

    """
    studio_api = True
    url = get_api_url(protocol=protocol, host=host, port=None, endpoint=f"projects/{slug}", usr_api=studio_api)
    headers = {}

    _token = get_token(token, False)

    if _token:
        headers["Authorization"] = _token
    try:
        response = requests.get(url, headers=headers)
        response_json = response.json()
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url}")

    if response_json.get("error"):
        click.secho(f"No project with slug '{slug}' exists.", fg="red")
    else:
        print_response(response, "project", True)


@click.option("-s", "--slug", required=True, help="Slug name of project.")
@click.option("-p", "--protocol", required=False, default=STUDIO_DEFAULTS["protocol"], help="Communication protocol of studio (api)")
@click.option("-H", "--host", required=False, default=STUDIO_DEFAULTS["host"], help="Hostname of studio (api)")
@project_cmd.command("activate")
@click.pass_context
def set_active_project(ctx, slug: str = None, protocol: str = None, host: str = None):
    """Set active project.

    :param ctx:
    :param slug:
    """
    activate_project(slug, protocol, host)


def activate_project(slug: str = None, protocol: str = None, host: str = None):
    studio_api = True
    url_projects = get_api_url(protocol=protocol, host=host, port=None, endpoint="projects", usr_api=studio_api)
    url_project_token = get_api_url(protocol=protocol, host=host, port=None, endpoint="admin-token", usr_api=studio_api)

    context_path = os.path.join(home_dir, ".fedn")
    try:
        with open(f"{context_path}/context.yaml", "r") as yaml_file:
            context_data = yaml.safe_load(yaml_file)
    except Exception as e:
        print(f"Error: Failed to read YAML file. Details: {e}")

    user_access_token = context_data.get("User tokens").get("access")
    _token = get_token(user_access_token, True)
    headers_projects = {}

    if _token:
        headers_projects["Authorization"] = _token

    try:
        response_projects = requests.get(url_projects, headers=headers_projects)
        projects_response_json = response_projects.json()
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url_projects}")
    if len(projects_response_json) > 0:
        if slug is None:
            headers_projects["X-Project-Slug"] = projects_response_json[0].get("slug")
            slug = projects_response_json[0].get("slug")
        else:
            for i in projects_response_json:
                if i.get("slug") == slug:
                    headers_projects["X-Project-Slug"] = i.get("slug")

        controller_url = f"{protocol}://{host}/{slug}-fedn-reducer"

        try:
            response_project_tokens = requests.get(url_project_token, headers=headers_projects)
        except requests.exceptions.ConnectionError:
            click.echo(f"Error: Could not connect to {url_project_token}")

        project_tokens = response_project_tokens.json()
        context_data["Active project tokens"] = project_tokens
        context_data["Active project slug"] = slug
        context_data["Active project url"] = controller_url

        try:
            with open(f"{context_path}/context.yaml", "w") as yaml_file:
                yaml.dump(context_data, yaml_file, default_flow_style=False)  # Add access and refresh tokens to context yaml file
        except Exception as e:
            print(f"Error: Failed to write to YAML file. Details: {e}")

        click.secho(f"Project with slug {slug} is now active.", fg="green")
