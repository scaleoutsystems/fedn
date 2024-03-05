import os
from enum import Enum

import yaml

global STATESTORE_CONFIG
global MODELSTORAGE_CONFIG


class StorageType(Enum):
    STATESTORE_CONFIG = 1,
    MODELSTORAGE_CONFIG = 2


fedn_config = {
    "statestore": {
        "type": "MongoDB",
        "mongo_config": {
            "host": "localhost",
            "port": 27017
        }
    },
    "network_id": "fedn-network",
    "controller": {
        "host": "localhost",
        "port": 8092,
        "debug": True
    },
    "storage": {
        "storage_type": "filesystem",
        "storage_config": {
            "storage_path": "/tmp/fedn/files"
        }
    }
}


def get_environment_config():
    """ Get the configuration from environment variables.
    """
    global STATESTORE_CONFIG
    global MODELSTORAGE_CONFIG

    STATESTORE_CONFIG = os.environ.get('STATESTORE_CONFIG', None)
    MODELSTORAGE_CONFIG = os.environ.get('MODELSTORAGE_CONFIG', None)


def get_config(file: str = None, storage_type: StorageType = StorageType.STATESTORE_CONFIG) -> dict:
    """ Get the configuration from file.
    If file is not provided, the environment variables are checked for the configuration file paths.
    If the environment variables are not set, the default configuration is used (fedn_config).

    :param file: The configuration file (yaml) path (optional).
    :type file: str
    :return: The configuration as a dict.
    :rtype: dict
    """
    settings: dict = None

    if file is None:
        get_environment_config()
        file = STATESTORE_CONFIG if storage_type == StorageType.STATESTORE_CONFIG else MODELSTORAGE_CONFIG

    if file:
        with open(file, 'r') as config_file:
            try:
                settings = dict(yaml.safe_load(config_file))
            except yaml.YAMLError as e:
                raise (e)

    return settings or fedn_config


def get_statestore_config(file: str = None):
    """ Get the statestore configuration from file.

    :param file: The statestore configuration file (yaml) path (optional).
    :type file: str
    :return: The statestore configuration as a dict.
    :rtype: dict
    """
    settings: dict = get_config(file, StorageType.STATESTORE_CONFIG)

    return settings["statestore"]


def get_modelstorage_config(file=None):
    """ Get the model storage configuration from file.

    :param file: The model storage configuration file (yaml) path (optional).
    :type file: str
    :return: The model storage configuration as a dict.
    :rtype: dict
    """
    settings: dict = get_config(file, StorageType.MODELSTORAGE_CONFIG)

    return settings["storage"]


def get_network_config(file=None):
    """ Get the network configuration from file.

    :param file: The network configuration file (yaml) path (optional).
    :type file: str
    :return: The network id.
    :rtype: str
    """
    settings: dict = get_config(file, StorageType.STATESTORE_CONFIG)

    return settings["network_id"]


def get_controller_config(file=None):
    """ Get the controller configuration from file.

    :param file: The controller configuration file (yaml) path (optional).
    :type file: str
    :return: The controller configuration as a dict.
    :rtype: dict
    """
    settings: dict = get_config(file, StorageType.STATESTORE_CONFIG)

    return settings["controller"]
