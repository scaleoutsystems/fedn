import yaml

from scaleoututil.logging import ScaleoutLogger


def read_yaml_file(file_path):
    try:
        cfg = None
        with open(file_path, "rb") as config_file:
            cfg = yaml.safe_load(config_file.read())

    except Exception as e:
        ScaleoutLogger().error(f"Error trying to read yaml file: {file_path}")
        raise e
    return cfg
