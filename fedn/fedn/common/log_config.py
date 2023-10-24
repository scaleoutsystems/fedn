import logging
import logging.config

import urllib3

from fedn.common.color_handler import ColorizingStreamHandler

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3").setLevel(logging.ERROR)

handler = ColorizingStreamHandler(theme='dark')
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


def set_theme_from_string(theme_str):
    """
    Set the logging color theme based on a string input.
    """
    # Check if the theme string is valid
    valid_themes = ['dark', 'light', 'default']
    if theme_str.lower() not in valid_themes:
        raise ValueError(f"Invalid theme: {theme_str}. Valid themes are: {', '.join(valid_themes)}")

    # Set the theme for the ColorizingStreamHandler
    handler.set_theme(theme_str.lower())
