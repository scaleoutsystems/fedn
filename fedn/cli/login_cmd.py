import os
from getpass import getpass

import click
import requests
import yaml

from .main import main

# Replace this with the platform's actual login endpoint
home_dir = os.path.expanduser("~")


@main.group("studio")
@click.pass_context
def login_cmd(ctx):
    """:param ctx:"""
    pass


@login_cmd.command("login")
@click.option("-p", "--protocol", required=False, default="https", help="Communication protocol")
@click.option("-H", "--host", required=False, default="fedn.scaleoutsystems.com", help="Hostname of controller (api)")
@click.pass_context
def login_cmd(ctx, protocol: str, host: str):
    """Logging into FEDn Studio"""
    # Step 1: Display welcome message
    click.secho("Welcome to Scaleout FEDn!", fg="green")

    url = f"{protocol}://{host}/api/token/"

    # Step 3: Prompt for username and password
    username = input("Please enter your username: ")
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
        data = response.json()
        if data.get("access"):
            click.secho("Login successful!", fg="green")
            context_path = os.path.join(home_dir, ".fedn")
            if not os.path.exists(context_path):
                os.makedirs(context_path)
            try:
                with open(f"{context_path}/context.yaml", "w") as yaml_file:
                    yaml.dump(data, yaml_file, default_flow_style=False)  # Add access and refresh tokens to context yaml file
            except Exception as e:
                print(f"Error: Failed to write to YAML file. Details: {e}")
        else:
            click.secho("Login failed. Please check your credentials.", fg="red")
    else:
        click.secho(f"Unexpected error: {response.text}", fg="red")
