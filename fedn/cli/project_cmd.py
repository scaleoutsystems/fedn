import os

import click
import requests

from .main import main
from .shared import HOME_DIR, STUDIO_DEFAULTS, get_api_url, get_context, get_response, get_token, print_response, set_context


@main.group("project")
@click.pass_context
def project_cmd(ctx):
    """:param ctx:"""
    pass


@click.option("-id", "--id", required=True, help="ID of project.")
@click.option("-p", "--protocol", required=False, default=STUDIO_DEFAULTS["protocol"], help="Communication protocol of studio (api)")
@click.option("-H", "--host", required=False, default=STUDIO_DEFAULTS["host"], help="Hostname of studio (api)")
@click.option("-y", "--yes", is_flag=True, help="Automatically confirm any prompts.")
@project_cmd.command("delete")
@click.pass_context
def delete_project(ctx, id: str = None, protocol: str = None, host: str = None, yes: bool = False):
    """Delete project with given ID."""
    # Check if project with given id exists
    studio_api = True

    response = get_response(protocol=protocol, host=host, port=None, endpoint=f"projects/{id}", token=None, headers={}, usr_api=studio_api, usr_token=False)
    if response.status_code == 200:
        if response.json().get("error"):
            click.secho(f"No project with id '{id}' exists.", fg="red")
        elif yes or input(f"Are you sure you want to delete project with id {id} (y/n)?: ").lower() == "y":
            url = get_api_url(protocol=protocol, host=host, port=None, endpoint=f"projects/delete/{id}", usr_api=studio_api)
            headers = {}

            _token = get_token(None, True)

            if _token:
                headers["Authorization"] = _token
            # Call the authentication API
            try:
                requests.delete(url, headers=headers)
                click.secho(f"Project with slug {id} has been removed.", fg="green")
            except requests.exceptions.RequestException as e:
                click.echo(str(e), fg="red")
            activate_project(None, protocol, host)
    else:
        click.secho(f"Unexpected error: {response.status_code}", fg="red")


@click.option("-n", "--name", required=False, default=None, help="Name of new project.")
@click.option("-d", "--description", required=False, default=None, help="Description of new project.")
@click.option("-p", "--protocol", required=False, default=STUDIO_DEFAULTS["protocol"], help="Communication protocol of studio (api)")
@click.option("-H", "--host", required=False, default=STUDIO_DEFAULTS["host"], help="Hostname of studio (api)")
@click.option("--no-interactive", is_flag=True, help="Run in non-interactive mode.")
@project_cmd.command("create")
@click.pass_context
def create_project(ctx, name: str = None, description: str = None, protocol: str = None, host: str = None, no_interactive: bool = False):
    """Create project.
    :param ctx:
    """
    # Check if user can create project
    studio_api = True
    url = get_api_url(protocol=protocol, host=host, port=None, endpoint="projects/create", usr_api=studio_api)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    _token = get_token(None, True)

    if _token:
        headers["Authorization"] = _token
    if not no_interactive:
        if name is None:
            name = input("Please enter a project name: ")
        if description is None:
            description = input("Please enter a project description (optional): ")
    else:
        if name is None:
            click.secho("Project name is required.", fg="red")
            return
        if description is None:
            description = ""
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
@project_cmd.command("list")
@click.pass_context
def list_projects(ctx, protocol: str = None, host: str = None):
    """Return:
    ------
    - result: list of projects

    """
    studio_api = True

    response = get_response(protocol=protocol, host=host, port=None, endpoint="projects", token=None, headers={}, usr_api=studio_api, usr_token=True)

    if response.status_code == 200:
        response_json = response.json()
        if len(response_json) > 0:
            context_path = os.path.join(HOME_DIR, ".fedn")
            context_data = get_context(context_path)
            active_project = context_data.get("Active project id")

            for i in response_json:
                project_name = i.get("slug")
                if project_name == active_project:
                    click.secho(f"{project_name} (active)", fg="green")
                else:
                    click.secho(project_name)
    else:
        click.secho(f"Unexpected error: {response.status_code}", fg="red")


@click.option("-id", "--id", required=True, help="ID of project.")
@click.option("-p", "--protocol", required=False, default=STUDIO_DEFAULTS["protocol"], help="Communication protocol of studio (api)")
@click.option("-H", "--host", required=False, default=STUDIO_DEFAULTS["host"], help="Hostname of studio (api)")
@project_cmd.command("get")
@click.pass_context
def get_project(ctx, id: str = None, protocol: str = None, host: str = None):
    """Return:
    ------
    - result: project with given id

    """
    studio_api = True

    response = get_response(protocol=protocol, host=host, port=None, endpoint=f"projects/{id}", token=None, headers={}, usr_api=studio_api, usr_token=False)

    if response.status_code == 200:
        response_json = response.json()

        if response_json.get("error"):
            click.secho(f"No project with id '{id}' exists.", fg="red")
        else:
            print_response(response, "project", True)
    else:
        click.secho(f"Unexpected error: {response.status_code}", fg="red")


@click.option("-id", "--id", required=True, default=None, help="Name of new project.")
@click.option("-p", "--protocol", required=False, default=STUDIO_DEFAULTS["protocol"], help="Communication protocol of studio (api)")
@click.option("-H", "--host", required=False, default=STUDIO_DEFAULTS["host"], help="Hostname of studio (api)")
@project_cmd.command("update")
@click.pass_context
def update_project(ctx, id: str = None, protocol: str = None, host: str = None):
    """Update project to latest version.
    :param ctx:
    """
    # Check if user can create project
    studio_api = True

    response = get_response(protocol=protocol, host=host, port=None, endpoint=f"projects/{id}", token=None, headers={}, usr_api=studio_api, usr_token=False)
    if response.status_code == 200:
        if response.json().get("error"):
            click.secho(f"No project with id '{id}' exists.", fg="red")
        else:
            url = get_api_url(protocol=protocol, host=host, port=None, endpoint="projects/update", usr_api=studio_api)
            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            _token = get_token(None, True)

            if _token:
                headers["Authorization"] = _token

            # Call the authentication API
            try:
                requests.post(url, data={"slug": id}, headers=headers)
            except requests.exceptions.RequestException as e:
                click.secho(str(e), fg="red")
            click.secho(f"Project with id '{id}' is up-to-date.", fg="green")
    else:
        click.secho(f"Unexpected error: {response.status_code}", fg="red")


@click.option("-id", "--id", required=True, help="id name of project.")
@click.option("-p", "--protocol", required=False, default=STUDIO_DEFAULTS["protocol"], help="Communication protocol of studio (api)")
@click.option("-H", "--host", required=False, default=STUDIO_DEFAULTS["host"], help="Hostname of studio (api)")
@project_cmd.command("set-context")
@click.pass_context
def set_active_project(ctx, id: str = None, protocol: str = None, host: str = None):
    """Set active project.

    :param ctx:
    :param id:
    """
    activate_project(id, protocol, host)


def activate_project(id: str = None, protocol: str = None, host: str = None):
    """Sets project with give ID as active by updating context file."""
    studio_api = True
    headers_projects = {}
    context_path = os.path.join(HOME_DIR, ".fedn")
    context_data = get_context(context_path)

    user_access_token = context_data.get("User tokens").get("access")

    response_projects = get_response(
        protocol=protocol, host=host, port=None, endpoint="projects", token=user_access_token, headers=headers_projects, usr_api=studio_api, usr_token=False
    )
    if response_projects.status_code == 200:
        projects_response_json = response_projects.json()
        if len(projects_response_json) > 0:
            if id is None:
                headers_projects["X-Project-Slug"] = projects_response_json[0].get("slug")
                id = projects_response_json[0].get("slug")
            else:
                project_found = False
                for i in projects_response_json:
                    if i.get("slug") == id:
                        project_found = True
                        headers_projects["X-Project-Slug"] = i.get("slug")
                if not project_found:
                    click.secho(f"No project found with id {id}", fg="red")
                    return
            controller_url = f"{protocol}://{host}/{id}-fedn-reducer"

            response_project_tokens = get_response(
                protocol=protocol,
                host=host,
                port=None,
                endpoint="admin-token",
                token=user_access_token,
                headers=headers_projects,
                usr_api=studio_api,
                usr_token=False,
            )
            if response_project_tokens.status_code == 200:
                project_tokens = response_project_tokens.json()
                context_data["Active project tokens"] = project_tokens
                context_data["Active project id"] = id
                context_data["Active project url"] = controller_url

                set_context(context_path, context_data)

                click.secho(f"Project with slug {id} is now active.", fg="green")
            else:
                click.secho(f"Unexpected error: {response_project_tokens.status_code}", fg="red")
        else:
            click.echo("No projects available to set current context.")
    else:
        click.secho(f"Unexpected error: {response_projects.status_code}", fg="red")

def no_project_exists(response) -> bool:
    """Returns true if no project exists."""
    response_json = response.json()
    print(response_json)
    if type(response_json) is list:
        return False
    elif response_json.get("error"):
        return True
    return False
