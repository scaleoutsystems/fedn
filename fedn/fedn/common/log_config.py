import logging
import logging.config

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3").setLevel(logging.ERROR)

handler = logging.StreamHandler()
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)


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
