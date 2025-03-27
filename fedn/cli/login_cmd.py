import os
from getpass import getpass

import click
import requests

from .main import main
from .shared import HOME_DIR, STUDIO_DEFAULTS, get_response, set_context


@main.group("studio")
@click.pass_context
def login_cmd(ctx):
    """:param ctx:"""
    pass


@login_cmd.command("login")
@click.option("-u", "--username", required=False, default=None, help="username in studio")
@click.option("-P", "--password", required=False, default=None, help="password in studio")
@click.option("-p", "--protocol", required=False, default=STUDIO_DEFAULTS["protocol"], help="Communication protocol of studio (api)")
@click.option("-H", "--host", required=False, default=STUDIO_DEFAULTS["host"], help="Hostname of studio (api)")
@click.pass_context
def login_cmd(ctx, protocol: str, host: str, username: str, password: str):
    """Login to FEDn Studio"""
    # Step 1: Display welcome message
    click.secho("Welcome to Scaleout FEDn!", fg="green")

    url = f"{protocol}://{host}/api/token/"

    # Step 3: Prompt for username and password
    if username is None:
        username = input("Please enter your username: ")
    if password is None:
        password = getpass("Please enter your password: ")

    # Call the authentication API
    try:
        response = requests.post(url, json={"username": username, "password": password}, headers={"Content-Type": "application/json"})
        response.raise_for_status()  # Raise an error for HTTP codes 4xx/5xx
    except requests.exceptions.RequestException as e:
        click.secho("Error connecting to the platform. Please try again.", fg="red")
        click.secho(str(e), fg="red")
        return

    # Handle the response
    if response.status_code == 200:
        context_data = get_context(response, protocol, host)

        context_path = os.path.join(HOME_DIR, ".fedn")
        if not os.path.exists(context_path):
            os.makedirs(context_path)
        set_context(context_path, context_data)
    else:
        click.secho(f"Unexpected error: {response.status_code}", fg="red")


# Sets the context for a given user
def get_context(response, protocol: str, host: str):
    """Generates content for context file with the following data:
    User tokens: access and refresh token to authenticate user towards Studio
    Active project tokens: access and refresh token to authenticate user towards controller
    Active project id: slug of active project
    Active project url: controller url of active project
    """
    context_data = {"User tokens": {}, "Active project tokens": {}, "Active project id": {}, "Active project url": {}}
    user_token_data = response.json()
    if user_token_data.get("access"):
        context_data["User tokens"] = user_token_data
        studio_api = True
        headers_projects = {}
        user_access_token = user_token_data.get("access")
        response_projects = get_response(
            protocol=protocol,
            host=host,
            port=None,
            endpoint="projects",
            token=user_access_token,
            headers=headers_projects,
            usr_api=studio_api,
            usr_token=True,
        )
        if response_projects.status_code == 200:
            projects_response_json = response_projects.json()
            if len(projects_response_json) > 0:
                id = projects_response_json[0].get("slug")
                context_data["Active project id"] = id
                headers_projects["X-Project-Slug"] = id
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
                    controller_url = f"{protocol}://{host}/{id}-fedn-reducer"
                    context_data["Active project url"] = controller_url
                else:
                    click.secho(f"Unexpected error: {response_project_tokens.status_code}", fg="red")
        else:
            click.secho(f"Unexpected error: {response_projects.status_code}", fg="red")
        click.secho("Login successful!", fg="green")
    else:
        click.secho("Login failed. Please check your credentials.", fg="red")

    return context_data
