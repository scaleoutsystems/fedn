import os
from getpass import getpass

import click
import requests
import yaml

from .main import main
from .shared import STUDIO_DEFAULTS, get_api_url, get_token

# Replace this with the platform's actual login endpoint
home_dir = os.path.expanduser("~")


@main.group("studio")
@click.pass_context
def login_cmd(ctx):
    """:param ctx:"""
    pass


@login_cmd.command("login")
@click.option("-", "--name", required=False, default=None, help="username in studio")
@click.option("-pw", "--password", required=False, default=None, help="password in studio")
@click.option("-p", "--protocol", required=False, default=STUDIO_DEFAULTS["protocol"], help="Communication protocol of studio (api)")
@click.option("-H", "--host", required=False, default=STUDIO_DEFAULTS["host"], help="Hostname of studio (api)")
@click.pass_context
def login_cmd(ctx, protocol: str, host: str, username: str, password: str):
    """Logging into FEDn Studio"""
    # Step 1: Display welcome message
    click.secho("Welcome to Scaleout FEDn!", fg="green")

    url = f"{protocol}://{host}/api/token/"

    # Step 3: Prompt for username and password
    if username is None and password is None:
        username = input("Please enter your username: ")
        password = getpass("Please enter your password: ")
    elif password is None:
        password = getpass("Please enter your password: ")
    else:
        username = input("Please enter your username: ")

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

        context_path = os.path.join(home_dir, ".fedn")
        if not os.path.exists(context_path):
            os.makedirs(context_path)
        try:
            with open(f"{context_path}/context.yaml", "w") as yaml_file:
                yaml.dump(context_data, yaml_file, default_flow_style=False)  # Add access and refresh tokens to context yaml file
        except Exception as e:
            print(f"Error: Failed to write to YAML file. Details: {e}")
    else:
        click.secho(f"Unexpected error: {response.text}", fg="red")


def get_context(response, protocol, host):
    user_token_data = response.json()
    if user_token_data.get("access"):
        studio_api = True
        url_projects = get_api_url(protocol=protocol, host=host, port=None, endpoint="projects", usr_api=studio_api)
        headers_projects = {}
        user_access_token = user_token_data.get("access")
        _token = get_token(user_access_token, True)
        if _token:
            headers_projects["Authorization"] = _token

        try:
            response_projects = requests.get(url_projects, headers=headers_projects)
            projects_response_json = response_projects.json()
        except requests.exceptions.ConnectionError:
            click.echo(f"Error: Could not connect to {url_projects}")

        slug = projects_response_json[0].get("slug")
        headers_projects["X-Project-Slug"] = slug
        url_project_token = get_api_url(protocol=protocol, host=host, port=None, endpoint="admin-token", usr_api=studio_api)
        print(url_project_token)
        try:
            response_project_tokens = requests.get(url_project_token, headers=headers_projects)
            project_tokens = response_project_tokens.json()
        except requests.exceptions.ConnectionError:
            click.echo(f"Error: Could not connect to {url_project_token}")

        controller_url = f"{protocol}://{host}/{slug}-fedn-reducer"
        context_data = {
            "User tokens": user_token_data,
            "Active project tokens": project_tokens,
            "Active project slug": slug,
            "Active project url": controller_url,
        }
        click.secho("Login successful!", fg="green")
        return context_data
    else:
        click.secho("Login failed. Please check your credentials.", fg="red")
