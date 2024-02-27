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

def get_boolean_env(key, default=False):
    """ Get environment variable as boolean.

    :param key: The environment variable key.
    :type key: str
    :param default: The default value if the environment variable is not set (optional).
    :type default: bool
    :return: The environment variable value.
    :rtype: bool
    """
    value = os.environ.get(key, default)
    if isinstance(value, str):
        return value.lower() in ['true', '1']
    return value


def get_environment_config():
    """ Get the configuration from environment variables.
    """
    global STATESTORE_CONFIG
    global MODELSTORAGE_CONFIG
    global COMBINER_CONFIG
    global CONTROLLER_CONFIG

    STATESTORE_CONFIG = os.environ.get('FEDN_STATESTORE_CONFIG',
                                       '/workspaces/fedn/config/settings-controller.yaml.template')
    MODELSTORAGE_CONFIG = os.environ.get('FEDN_MODELSTORAGE_CONFIG',
                                         '/workspaces/fedn/config/settings-controller.yaml.template')
    COMBINER_CONFIG = os.environ.get('FEDN_COMBINER_CONFIG',
                                     '/workspaces/fedn/config/settings-combiner.yaml.template')
    CONTROLLER_CONFIG = os.environ.get('FEDN_CONTROLLER_CONFIG',
                                       '/workspaces/fedn/config/settings-controller.yaml.template')

def read_config_file(file):
    """ Read a yaml configuration file.

    :param file: The configuration file path.
    :type file: str
    :return: The configuration as a dict.
    :rtype: dict
    """
    with open(file, 'r') as config_file:
        try:
            config = dict(yaml.safe_load(config_file))
        except yaml.YAMLError as e:
            raise (e)
    return config

def get_statestore_config(file=None):
    """ Get the statestore configuration.

    :param file: The statestore configuration file (yaml) path (optional).
    :type file: str
    :return: The statestore configuration as a dict.
    :rtype: dict
    """
    if file is None:
        # check if environment variables are set
        host = get_env("FEDN_STATESTORE_HOST")
        if host is not None:
            return {
                "type": "MongoDB",
                "mongo_config": {
                    "host": host,
                    "port": int(get_env("FEDN_STATESTORE_PORT", 6534)),
                    "username": get_env("FEDN_STATESTORE_USER", "fedn_admin"),
                    "password": get_env("FEDN_STATESTORE_PASSWORD"),
                }
            }
        else:
            # else use the default file
            get_environment_config()
            file = STATESTORE_CONFIG
    settings = read_config_file(file)
    return settings["statestore"]


def get_modelstorage_config(file=None):
    """ Get the model storage configuration.

    :param file: The model storage configuration file (yaml) path (optional).
    :type file: str
    :return: The model storage configuration as a dict.
    :rtype: dict
    """
    if file is None:
        # check if environment variables are set
        host = get_env("FEDN_MODELSTORAGE_HOST")
        if host is not None:
            return {
                "storage_type": "S3",
                "storage_config": {
                    "storage_hostname": host,
                    "storage_port": int(get_env("FEDN_MODELSTORAGE_PORT", 9000)),
                    "storage_access_key": get_env("FEDN_MODELSTORAGE_ACCESS_KEY"),
                    "storage_secret_key": get_env("FEDN_MODELSTORAGE_SECRET_KEY"),
                    "storage_bucket": get_env("FEDN_MODELSTORAGE_BUCKET", "fedn-models"),
                    "context_bucket": get_env("FEDN_MODELSTORAGE_CONTEXT_BUCKET", "fedn-context"),
                    "storage_secure_mode": get_boolean_env("FEDN_MODELSTORAGE_SECURE_MODE", False),
                }
            }
        else:
            # else use the config file
            get_environment_config()
            file = MODELSTORAGE_CONFIG
    settings = read_config_file(file)
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
            # TODO: This is a temporary fix, network_id is the database name in the statestore
            file = STATESTORE_CONFIG
    settings = read_config_file(file)
    return settings["network_id"]


def get_controller_config(file=None):
    """ Get the controller configuration.

    :param file: The controller configuration file (yaml) path (optional).
    :type file: str
    :return: The controller configuration as a dict.
    :rtype: dict
    """
    if file is None:
        # check if environment variables are set
        host = get_env("FEDN_CONTROLLER_HOST")
        if host is not None:
            return {
                "host": host,
                "port": int(get_env("FEDN_CONTROLLER_PORT", 8092)),
                "debug": get_boolean_env("FEDN_CONTROLLER_DEBUG", False),
            }
        else:
            # else use the default file
            get_environment_config()
            file = CONTROLLER_CONFIG
    settings = read_config_file(file)
    return settings["controller"]

def get_combiner_config(file=None):
    """ Get the combiner configuration.

    :param file: The combiner configuration file (yaml) path (optional).
    :type file: str
    :return: The combiner configuration as a dict.
    :rtype: dict
    """
    if file is None:
        # check if environment variables are set
        host = get_env("FEDN_COMBINER_HOST")
        if host is not None:
            return {
                "host": host,
                "name": get_env("FEDN_COMBINER_NAME"),
                "port": int(get_env("FEDN_COMBINER_PORT", 12080)),
                "max_clients": int(get_env("FEDN_COMBINER_MAX_CLIENTS", 30)),
                "network_id": get_network_config(),
                "discovery_host": get_env("FEDN_CONTROLLER_HOST"),
                "discovery_port": int(get_env("FEDN_CONTROLLER_PORT", 8092)),
                "fqdn": get_env("FEDN_COMBINER_FQDN"),
                "secure": get_boolean_env("FEDN_GRPC_SECURE", False),
                "verify": get_boolean_env("FEDN_VERIFY_TLS", False),
            }
        else:
            # else use the default file
            get_environment_config()
            file = COMBINER_CONFIG
    settings = read_config_file(file)
    return settings["combiner"]