
import yaml

from fedn.common.log_config import logger


def apply_config(config):
    """Parse client config from file.

    Override configs from the CLI with settings in config file.

    :param config: Client config (dict).
    """
    with open(config['init'], 'r') as file:
        try:
            settings = dict(yaml.safe_load(file))
        except Exception:
            logger.error('Failed to read config from settings file, exiting.')
            return

    for key, val in settings.items():
        config[key] = val