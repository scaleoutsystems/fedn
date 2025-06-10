"""Shared objects for the network API."""

import os
from typing import Tuple

from flask import g
from werkzeug.security import safe_join

from fedn.network.common.network import Network
from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.s3.repository import Repository
from fedn.utils.checksum import sha


def get_db() -> DatabaseConnection:
    """Get the database connection."""
    if "db" not in g:
        raise RuntimeError("Database connection not initialized. Call start_server_api() first.")
    return g.db


def get_repository() -> Repository:
    """Get the repository."""
    if "repository" not in g:
        raise RuntimeError("Repository not initialized. Call start_server_api() first.")
    return g.repository


def get_network() -> Network:
    """Get the network interface."""
    if "network" not in g:
        raise RuntimeError("Network not initialized. Call start_server_api() first.")
    return g.network


def get_checksum(name: str = None) -> Tuple[bool, str, str]:
    """Generate a checksum for a given file."""
    message = None
    sum = None
    success = False

    if name is None:
        db = get_db().db
        active_package = db.package_store.get_active()
        if active_package is None:
            message = "No compute package uploaded"
            return success, message, sum
        name = active_package.storage_file_name
    file_path = safe_join(os.getcwd(), name)
    try:
        sum = str(sha(file_path))
        success = True
        message = "Checksum created."
    except FileNotFoundError:
        message = "File not found."
    return success, message, sum
