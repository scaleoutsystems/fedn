import os

import yaml

from fedn.utils.dist import get_package_path

SECRET_KEY = os.environ.get("FEDN_JWT_SECRET_KEY", False)
FEDN_JWT_CUSTOM_CLAIM_KEY = os.environ.get("FEDN_JWT_CUSTOM_CLAIM_KEY", False)
FEDN_JWT_CUSTOM_CLAIM_VALUE = os.environ.get("FEDN_JWT_CUSTOM_CLAIM_VALUE", False)

FEDN_AUTH_WHITELIST_URL_PREFIX = os.environ.get("FEDN_AUTH_WHITELIST_URL_PREFIX", False)
FEDN_JWT_ALGORITHM = os.environ.get("FEDN_JWT_ALGORITHM", "HS256")
FEDN_AUTH_SCHEME = os.environ.get("FEDN_AUTH_SCHEME", "Bearer")
FEDN_AUTH_REFRESH_TOKEN_URI = os.environ.get("FEDN_AUTH_REFRESH_TOKEN_URI", False)
FEDN_AUTH_REFRESH_TOKEN = os.environ.get("FEDN_AUTH_REFRESH_TOKEN", False)
FEDN_CUSTOM_URL_PREFIX = os.environ.get("FEDN_CUSTOM_URL_PREFIX", "")


FEDN_PACKAGE_EXTRACT_DIR = os.environ.get("FEDN_PACKAGE_EXTRACT_DIR", "package")

FEDN_COMPUTE_PACKAGE_DIR = os.environ.get("FEDN_COMPUTE_PACKAGE_DIR", "/app/client/package/")


def get_environment_config():
    """Get the configuration from environment variables."""
    global STATESTORE_CONFIG
    global MODELSTORAGE_CONFIG
    if not os.environ.get("STATESTORE_CONFIG", False):
        STATESTORE_CONFIG = get_package_path() + "/common/settings-controller.yaml.template"
    else:
        STATESTORE_CONFIG = os.environ.get("STATESTORE_CONFIG")

    if not os.environ.get("MODELSTORAGE_CONFIG", False):
        MODELSTORAGE_CONFIG = get_package_path() + "/common/settings-controller.yaml.template"
    else:
        MODELSTORAGE_CONFIG = os.environ.get("MODELSTORAGE_CONFIG")


def get_statestore_config(file=None):
    """Get the statestore configuration from file.

    :param file: The statestore configuration file (yaml) path (optional).
    :type file: str
    :return: The statestore configuration as a dict.
    :rtype: dict
    """
    if file is None:
        get_environment_config()
        file = STATESTORE_CONFIG
    with open(file, "r") as config_file:
        try:
            settings = dict(yaml.safe_load(config_file))
        except yaml.YAMLError as e:
            raise (e)
    return settings["statestore"]


def get_modelstorage_config(file=None):
    """Get the model storage configuration from file.

    :param file: The model storage configuration file (yaml) path (optional).
    :type file: str
    :return: The model storage configuration as a dict.
    :rtype: dict
    """
    if file is None:
        get_environment_config()
        file = MODELSTORAGE_CONFIG
    with open(file, "r") as config_file:
        try:
            settings = dict(yaml.safe_load(config_file))
        except yaml.YAMLError as e:
            raise (e)
    return settings["storage"]


def get_network_config(file=None):
    """Get the network configuration from file.

    :param file: The network configuration file (yaml) path (optional).
    :type file: str
    :return: The network id.
    :rtype: str
    """
    if file is None:
        get_environment_config()
        file = STATESTORE_CONFIG
    with open(file, "r") as config_file:
        try:
            settings = dict(yaml.safe_load(config_file))
        except yaml.YAMLError as e:
            raise (e)
    return settings["network_id"]


def get_controller_config(file=None):
    """Get the controller configuration from file.

    :param file: The controller configuration file (yaml) path (optional).
    :type file: str
    :return: The controller configuration as a dict.
    :rtype: dict
    """
    if file is None:
        get_environment_config()
        file = STATESTORE_CONFIG
    with open(file, "r") as config_file:
        try:
            settings = dict(yaml.safe_load(config_file))
        except yaml.YAMLError as e:
            raise (e)
    return settings["controller"]
