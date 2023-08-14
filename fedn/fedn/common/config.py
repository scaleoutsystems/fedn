import os

import yaml

STATESTORE_CONFIG = os.environ.get('STATESTORE_CONFIG', '/workspaces/fedn/config/settings-reducer.yaml.template')
MODELSTORAGE_CONFIG = os.environ.get('MODELSTORAGE_CONFIG', '/workspaces/fedn/config/settings-reducer.yaml.template')


def get_statestore_config(file=STATESTORE_CONFIG):
    """ Get the statestore configuration from file.

    :param file: The statestore configuration file (yaml) path.
    :type file: str
    :return: The statestore configuration as a dict.
    :rtype: dict
    """
    with open(file, 'r') as config_file:
        try:
            settings = dict(yaml.safe_load(config_file))
        except yaml.YAMLError as e:
            raise (e)
    return settings["statestore"]


def get_modelstorage_config(file=MODELSTORAGE_CONFIG):
    """ Get the model storage configuration from file.

    :param file: The model storage configuration file (yaml) path.
    :type file: str
    :return: The model storage configuration as a dict.
    :rtype: dict
    """
    with open(file, 'r') as config_file:
        try:
            settings = dict(yaml.safe_load(config_file))
        except yaml.YAMLError as e:
            raise (e)
    return settings["storage"]


def get_network_config(file=STATESTORE_CONFIG):
    """ Get the network configuration from file.

    :param file: The network configuration file (yaml) path.
    :type file: str
    :return: The network id.
    :rtype: str
    """
    with open(file, 'r') as config_file:
        try:
            settings = dict(yaml.safe_load(config_file))
        except yaml.YAMLError as e:
            raise (e)
    return settings["network_id"]
