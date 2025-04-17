"""Shared objects for the network API."""

import os
from typing import Tuple

from werkzeug.security import safe_join

from fedn.network.controller.control import Control
from fedn.utils.checksum import sha


def get_checksum(name: str = None) -> Tuple[bool, str, str]:
    """Generate a checksum for a given file."""
    message = None
    sum = None
    success = False

    if name is None:
        db = Control.instance().db
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
