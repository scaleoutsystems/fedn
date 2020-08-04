import io
import logging
import os

import yaml

logger = logging.getLogger(__name__)


def ensure_empty(directory):
    if len(os.listdir(directory)) == 0:
        return True
    else:
        return False


def override_from_environment(data):
    import os

    for key, value in os.environ.items():
        if key.startswith("FEDN") or key.startswith("STACKN"):
            try:
                data[key.lower()] = value
            except KeyError:
                print("Error setting VALUE: {} for KEY: {}".format(value, key))

    """
    FEDN_ACCESS_KEY
    FEDN_MINIO_HOST
    FEDN_MINIO_PORT
    FEDN_CONTROLLER_PORT   ==> CONNECT_HOST
    FEDN_CONTROLLER_HOST   ==> CONNECT_HOST
    """

    return data


def load_config(filename):
    with open(filename, 'r') as stream:
        data = yaml.load(stream, Loader=yaml.SafeLoader)

        data = override_from_environment(data)
        return data


def save_config(data, filename):
    with io.open(filename, 'w', encoding='utf8') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)


def save_default_files(path):
    from pathlib import Path
    Path(path + '/train.py').touch()
    Path(path + '/validate.py').touch()
    Path(path + '/predict.py').touch()


def get_default_config_file_path():
    # In order of priority:
    # 1. Current folder/project.yaml
    # 2. ~/.scaleout/project.yaml
    config_file_path = os.getcwd() + '/project.yaml'
    if os.path.exists(config_file_path):
        return config_file_path
    home = os.path.expanduser("~")
    config_file_path = os.path.join(home, '.scaleout/project.yaml')
    if os.path.exists(config_file_path):
        return config_file_path
    return None


def save_default_config(path):
    entry_points = {
        'train': {'command': 'python3 train.py'},
        'validate': {'command': 'python3 validate.py'},
        'predict': {'command': 'python3 predict.py'},
    }

    data = {'Project': {
        'access_key': 'your_identification',
        'entry_points': entry_points
    },
        'Config': {
            'hosts': {'controller_host': 'localhost',
                      'controller_port': 12080,
                      'minio_host': 'localhost',
                      'minio_port': 9000}
        }
    }
    save_config(data, path + '/project.yaml')


def validate_config(filename):
    with open(filename, 'r') as stream:

        data = yaml.load(stream)

        statuses = []
        statuses.append('train' in data['Project']['entry_points'])
        statuses.append('validate' in data['Project']['entry_points'])

        # TODO add more validation

        for status in statuses:
            if status is False:
                return False

        return True


# TODO Move to test
if __name__ == '__main__':
    save_default_config('.')
    # print(load_config('project.yaml'))
    # print(validate_config('project.yaml'))
