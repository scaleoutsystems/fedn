import os

import yaml

global STATESTORE_CONFIG
global MODELSTORAGE_CONFIG

def get_env(key, default=None):
    """ Get environment variable.

    :param key: The environment variable key.
    :type key: str
    :param default: The default value if the environment variable is not set (optional).
    :type default: str
    :return: The environment variable value.
    :rtype: str
    """
    return os.environ.get(key, default)

def get_environment_config():
    """ Get the configuration from environment variables.
    """
    global STATESTORE_CONFIG
    global MODELSTORAGE_CONFIG

    STATESTORE_CONFIG = os.environ.get('STATESTORE_CONFIG',
                                       '/workspaces/fedn/config/settings-reducer.yaml.template')
    MODELSTORAGE_CONFIG = os.environ.get('MODELSTORAGE_CONFIG',
                                         '/workspaces/fedn/config/settings-reducer.yaml.template')


def get_statestore_config(file=None):
    """ Get the statestore configuration from file.

    :param file: The statestore configuration file (yaml) path (optional).
    :type file: str
    :return: The statestore configuration as a dict.
    :rtype: dict
    """
    if file is None:
        # check if environment variables are set
        host = get_env("STATESTORE_HOST", "mongo")
        if host is not None:
            return {
                "type": "MongoDB",
                "mongo_config": {
                    "host": host,
                    "port": int(get_env("STATESTORE_PORT", 6534)),
                    "username": get_env("STATESTORE_USER", "fedn_admin"),
                    "password": get_env("STATESTORE_PASSWORD"),
                }
            }
        else:
            # else use the default file
            get_environment_config()
            file = STATESTORE_CONFIG
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
        # check if environment variables are set
        host = get_env("MODELSTORAGE_HOST", "minio")
        if host is not None:
            return {
                "storage_type": "S3",
                "storage_config": {
                    "storage_hostname": host,
                    "storage_port": int(get_env("MODELSTORAGE_PORT", 9000)),
                    "storage_access_key": get_env("MODELSTORAGE_ACCESS_KEY"),
                    "storage_secret_key": get_env("MODELSTORAGE_SECRET_KEY"),
                    "storage_bucket": get_env("MODELSTORAGE_BUCKET", "fedn-models"),
                    "context_bucket": get_env("MODELSTORAGE_CONTEXT_BUCKET", "fedn-context"),
                    "storage_secure_mode": bool(get_env("MODELSTORAGE_SECURE_MODE", False)),
                }
            }
        else:
            # else use the default file
            get_environment_config()
            file = MODELSTORAGE_CONFIG
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
        # check if environment variables are set
        network_id = get_env("NETWORK_ID", "fedn-network")
        if network_id is not None:
            return network_id
        else:
            # else use the default file
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
