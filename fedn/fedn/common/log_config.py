import logging
import logging.config
import os
from functools import wraps

import requests
import urllib3

log_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3").setLevel(logging.ERROR)

handler = logging.StreamHandler()
logger = logging.getLogger("fedn")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)


class CustomHTTPHandler(logging.handlers.HTTPHandler):
    def __init__(self, host, url, method='POST', credentials=None, projectname='', apptype=''):
        super().__init__(host, url, method)
        self.credentials = credentials  # Basic auth (username, password)
        self.projectname = projectname
        self.apptype = apptype

    def emit(self, record):
        # Customize the log record, for example, by adding metadata
        # record.projectname = self.projectname
        # record.apptype = self.apptype

        # Convert log record to json format
        
        log_entry = self.mapLogRecord(record)

        log_entry = {
            "msg": log_entry['msg'],
            "levelname": log_entry['levelname'],
            "project": os.environ.get("PROJECT_ID"),
            "appinstance": os.environ.get("APP_ID")

        }
        # Setup headers
        headers = {
            'Content-type': 'application/json',
        }
        if self.credentials:
            # import base64
            # auth = base64.b64encode(f"{self.credentials[0]}:{self.credentials[1]}".encode('utf-8')).decode('utf-8')
            # headers['Authorization'] = f'Basic {auth}'
            headers['Authorization'] = f"Token {self.credentials[1]}"
        # Use http.client or requests to send the log data
        if self.method.lower() == 'post':
            requests.post(self.host+self.url, json=log_entry, headers=headers)
        else:
            # Implement other methods if needed, e.g., GET
            pass


# Remote logging can only be configured via environment variables for now.
REMOTE_LOG_SERVER = os.environ.get('FEDN_REMOTE_LOG_SERVER', False)
REMOTE_LOG_PATH = os.environ.get('FEDN_REMOTE_LOG_PATH', False)
REMOTE_LOG_LEVEL = os.environ.get('FEDN_REMOTE_LOG_LEVEL', 'INFO')

if REMOTE_LOG_SERVER:
    rloglevel = log_levels.get(REMOTE_LOG_LEVEL, logging.INFO)
    remote_username = os.environ.get('FEDN_REMOTE_LOG_USERNAME', False)
    remote_password = os.environ.get('FEDN_REMOTE_LOG_PASSWORD', False)
    if remote_username and remote_password:
        credentials = (remote_username, remote_password)
    else:
        credentials = None
    http_handler = CustomHTTPHandler(
        host=REMOTE_LOG_SERVER,
        url=REMOTE_LOG_PATH,
        method='POST',
        credentials=credentials,
        projectname='test-project',
        apptype='client'
    )
    http_handler.setLevel(rloglevel)
    logger.addHandler(http_handler)


def set_log_level_from_string(level_str):
    """
    Set the log level based on a string input.
    """
    # Mapping of string representation to logging constants
    level_mapping = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
    }

    # Get the logging level from the mapping
    level = level_mapping.get(level_str.upper())

    if not level:
        raise ValueError(f"Invalid log level: {level_str}")

    # Set the log level
    logger.setLevel(level)


def set_log_stream(log_file):
    """
    Redirect the log stream to a specified file, if log_file is set.
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
