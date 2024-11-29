import os

import click
import requests

from fedn.common.log_config import logger

from .main import main
from .shared import get_api_url


@main.group("project")
@click.pass_context
def project_cmd(ctx):
    """:param ctx:"""
    pass


@project_cmd.command("create")
@click.option("-n", "--name", required=True, help="Name of project")
@click.option("-d", "--description", required=False, help="Project description")
@click.pass_context
def create_project(ctx, name, description):
    """Create compute package.

    :param ctx:
    :param name:
    :param description:
    """
    return 0


@click.option("--n_max", required=False, help="Number of items to list")
@project_cmd.command("list")
@click.pass_context
def list_projects(ctx, n_max: int = None):
    """Return:
    ------
    - result: list of packages

    """
    url = get_api_url(protocol=protocol, host=host, port=port, endpoint="packages")
    headers = {}

    if n_max:
        headers["X-Limit"] = n_max

    _token = get_token(token)

    if _token:
        headers["Authorization"] = _token

    try:
        response = requests.get(url, headers=headers)
        print(response.json())
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url}")


@click.option("-n", "--name", required=True, help="Authentication token")
@project_cmd.command("get")
@click.pass_context
def get_project(ctx, name: str = None):
    """Return:
    ------
    - result: project with given name

    """
    # Define the endpoint and headers
    url = "http://localhost:8000/api/some-protected-view/"
    headers = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX3BrIjoxLCJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiY29sZF9zdHVmZiI6IuKYgyIsImV4cCI6MTIzNDU2LCJqdGkiOiJmZDJmOWQ1ZTFhN2M0MmU4OTQ5MzVlMzYyYmNhOGJjYSJ9.NHlztMGER7UADHZJlxNG0WSi22a2KaYSfd1S-AuT7lU"
    }

    # Make the GET request
    response = requests.get(url, headers=headers)

    # Print the response
    print(f"Status code: {response.status_code}")
    print(f"Response body: {response.text}")


@click.option("-n", "--name", required=True, help="Project name")
@project_cmd.command("set")
@click.pass_context
def set_project(ctx, name: str = None):
    """Return:
    ------

    """
    headers = {}

    _token = None

    if _token:
        headers["Authorization"] = _token

    try:
        response = requests.get(url, headers=headers)
        print(response.json())
    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to {url}")
