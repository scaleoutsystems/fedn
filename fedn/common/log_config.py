import logging
import logging.config
import os

import requests
import urllib3

log_levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3").setLevel(logging.ERROR)

handler = logging.StreamHandler()
logger = logging.getLogger("fedn")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)


class StudioHTTPHandler(logging.handlers.HTTPHandler):
    def __init__(self, host, url, method="POST", token=None):
        super().__init__(host, url, method)
        self.token = token

    def emit(self, record):
        log_entry = self.mapLogRecord(record)

        log_entry = {
            "msg": log_entry["msg"],
            "levelname": log_entry["levelname"],
            "project": os.environ.get("PROJECT_ID"),
            "appinstance": os.environ.get("APP_ID"),
        }
        # Setup headers
        headers = {
            "Content-type": "application/json",
        }
        if self.token:
            remote_token_protocol = os.environ.get("FEDN_REMOTE_LOG_TOKEN_PROTOCOL", "Token")
            headers["Authorization"] = f"{remote_token_protocol} {self.token}"
        if self.method.lower() == "post":
            requests.post(self.host + self.url, json=log_entry, headers=headers)
        else:
            # No other methods implemented.
            return


# Remote logging can only be configured via environment variables for now.
REMOTE_LOG_SERVER = os.environ.get("FEDN_REMOTE_LOG_SERVER", False)
REMOTE_LOG_PATH = os.environ.get("FEDN_REMOTE_LOG_PATH", False)
REMOTE_LOG_LEVEL = os.environ.get("FEDN_REMOTE_LOG_LEVEL", "INFO")

if REMOTE_LOG_SERVER:
    rloglevel = log_levels.get(REMOTE_LOG_LEVEL, logging.INFO)
    remote_token = os.environ.get("FEDN_REMOTE_LOG_TOKEN", None)

    http_handler = StudioHTTPHandler(host=REMOTE_LOG_SERVER, url=REMOTE_LOG_PATH, method="POST", token=remote_token)
    http_handler.setLevel(rloglevel)
    logger.addHandler(http_handler)


def set_log_level_from_string(level_str):
    """Set the log level based on a string input.
    """
    # Mapping of string representation to logging constants
    level_mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }

    # Get the logging level from the mapping
    level = level_mapping.get(level_str.upper())

    if not level:
        raise ValueError(f"Invalid log level: {level_str}")

    # Set the log level
    logger.setLevel(level)


def set_log_stream(log_file):
    """Redirect the log stream to a specified file, if log_file is set.
    """
    if not log_file:
        return

    # Remove existing handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # Create a FileHandler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)
