"""Shared objects for the network API."""

import os
from typing import Tuple

from werkzeug.security import safe_join

from fedn.network.common.network import Network
from fedn.network.storage.dbconnection import DatabaseConnection
from fedn.network.storage.s3.repository import Repository
from fedn.utils.checksum import sha


class ApplicationState:
    """Global state to hold shared objects for the network API."""

    _instance = None

    def _prepare(self):
        self.db = None
        self.repository = None
        self.network = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ApplicationState, cls).__new__(cls)
            cls._instance._prepare()
        return cls._instance


def get_db() -> DatabaseConnection:
    """Get the database connection."""
    return ApplicationState().db


def get_repository() -> Repository:
    """Get the repository."""
    return ApplicationState().repository


def get_network() -> Network:
    """Get the network interface."""
    return ApplicationState().network


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
