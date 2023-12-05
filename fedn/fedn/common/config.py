import os

import yaml

global STATESTORE_CONFIG
global MODELSTORAGE_CONFIG


def get_default_config():
    statestore_config = {
        "statestore": {
            "type": "MongoDB",
            "mongo_config": {
                "username": "admin",
                "password": "admin",
                "host": "localhost",
                "port": 27017
            }
        }
    }
    network_config = {
        "network_id": "fedn-network"
    }
    controller_config = {
        "controller": {
            "host": "localhost",
            "port": 8092,
            "debug": True
        }
    }
    storage_path = os.path.expanduser("~/.fedn/files")
    storage_config = {
        "storage": {
            "storage_type": "filesystem",
            "storage_config": {
                "storage_path": storage_path
            }
        }
    }
    fedn_config = {}
    fedn_config.update(statestore_config)
    fedn_config.update(network_config)
    fedn_config.update(controller_config)
    fedn_config.update(storage_config)
    return fedn_config


def get_environment_config():
    """ Get the configuration from environment variables.
    """
    global STATESTORE_CONFIG
    global MODELSTORAGE_CONFIG

    STATESTORE_CONFIG = os.environ.get('STATESTORE_CONFIG', None)
    MODELSTORAGE_CONFIG = os.environ.get('MODELSTORAGE_CONFIG', None)


def get_statestore_config(file=None):
    """ Get the statestore configuration from file.

    :param file: The statestore configuration file (yaml) path (optional).
    :type file: str
    :return: The statestore configuration as a dict.
    :rtype: dict
    """
    if file is None:
        get_environment_config()
    if STATESTORE_CONFIG:
        file = STATESTORE_CONFIG
    else:
        fedn_config = get_default_config()
        return fedn_config['statestore']

    with open(file, 'r') as config_file:
        try:
            settings = dict(yaml.safe_load(config_file))
        except yaml.YAMLError as e:
            raise (e)
    return settings["statestore"]


def get_modelstorage_config(file=None):
    """ Get the model storage configuration from file.

    :param file: The model storage configuration file (yaml) path (optional).
    :type file: str
    :return: The model storage configuration as a dict.
    :rtype: dict
    """
    if file is None:
        get_environment_config()

    if MODELSTORAGE_CONFIG:
        file = MODELSTORAGE_CONFIG
    else:
        fedn_config = get_default_config()
        return fedn_config['storage']

    with open(file, 'r') as config_file:
        try:
            settings = dict(yaml.safe_load(config_file))
        except yaml.YAMLError as e:
            raise (e)
    return settings["storage"]


def get_network_config(file=None):
    """ Get the network configuration from file.

    :param file: The network configuration file (yaml) path (optional).
    :type file: str
    :return: The network id.
    :rtype: str
    """
    if file is None:
        get_environment_config()
        file = STATESTORE_CONFIG
    with open(file, 'r') as config_file:
        try:
            settings = dict(yaml.safe_load(config_file))
        except yaml.YAMLError as e:
            raise (e)
    return settings["network_id"]


def get_controller_config(file=None):
    """ Get the controller configuration from file.

    :param file: The controller configuration file (yaml) path (optional).
    :type file: str
    :return: The controller configuration as a dict.
    :rtype: dict
    """
    if file is None:
        get_environment_config()
        file = STATESTORE_CONFIG
    with open(file, 'r') as config_file:
        try:
            settings = dict(yaml.safe_load(config_file))
        except yaml.YAMLError as e:
            raise (e)
    return settings["controller"]
