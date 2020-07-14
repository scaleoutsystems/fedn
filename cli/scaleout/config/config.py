import io
import yaml
import os

import logging

logger = logging.getLogger(__name__)


def ensure_empty(directory):
    if len(os.listdir(directory)) == 0:
        return True
    else:
        return False

def ensure_default_values(data):
    """ Set default values matching the local dev deployment of Studio. """
    if 'auth_url' not in data:
        data['auth_url'] = 'platform.local/api/api-token-auth'
    if 'username' not in data:
        data['username'] = 'testuser'
    if 'password' not in data:
        data['password'] = 'password'
    if 'so_domain_name' not in data:
        data['so_domain_name'] = 'local'
    logger.debug('setting default values where needed')

    return data

def ensure_alliance_default_values(data):
    if 'minio_host' not in data['Alliance']:
        data['Alliance']['minio_host'] = 'localhost'
    if 'minio_port' not in data['Alliance']:
        data['Alliance']['minio_port'] = '9000'
    if 'controller_host' not in data['Alliance']:
        data['Alliance']['controller_host'] = 'localhost'
    if 'controller_port' not in data['Alliance']:
        data['Alliance']['controller_port'] = '12080'
    logger.debug('setting default values where needed')

    return data


def override_from_environment(data):
    import os

    try:
        data['Project']['access_key'] = os.environ['ACCESS_KEY']
        print("setting access key!. {} {}".format(os.environ['ACCESS_KEY'], __file__))
    except KeyError:
        pass

    try:
        data['Config']['hosts']['minio_host'] = os.environ['MINIO_HOST']
    except KeyError:
        pass

    try:
        data['Config']['hosts']['minio_port'] = os.environ['MINIO_PORT']
    except KeyError:
        pass

    import os
    try:
        data['Config']['hosts']['controller_host'] = os.environ['CONTROLLER_HOST']
    except KeyError:
        pass

    import os
    try:
        data['Config']['hosts']['controller_port'] = os.environ['CONTROLLER_PORT']
    except KeyError:
        pass

    return data


def load_config(filename):
    with open(filename, 'r') as stream:
        data = yaml.load(stream,Loader=yaml.SafeLoader)
        data = ensure_default_values(data)
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
