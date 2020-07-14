import os
from scaleout.config.config import load_config as load_conf, get_default_config_file_path
from scaleout.repository.miniorepository import MINIORepository
from scaleout.errors import InvalidConfigurationError

class Project:
    project_dir = None
    config_file_path = None
    config = None

    def __init__(self, project_dir=None, config_file_path=None):
        if project_dir is None:
            self.project_dir = os.getcwd()
        else:
            self.project_dir = project_dir

        if config_file_path is None:
            self.config_file_path = get_default_config_file_path()
        else:
            self.config_file_path = config_file_path

        try:
            self._load_config()
        except Exception:
            raise InvalidConfigurationError("Missing config file")
        try:
            self.api_endpoint = os.path.join(self.config['so_domain_name'], '/api')
            self.auth_url = self.config["auth_url"]
            self.user = self.config['username']
            # self.project_name = self.config['Project']['project_name']
        except Exception:
            raise InvalidConfigurationError("Configuration file has missing values.")

    def _load_config(self, ):
        self.config = load_conf(self.config_file_path)

def init_project(project_dir):
    directory = None

    if project_dir is None:
        directory = os.getcwd()
    else:
        directory = os.getcwd() + '/' + project_dir

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

    if not ensure_empty(directory):
        print("BAILING: DIRECTORY IS NOT EMPTY!")
        return

    save_default_config(directory)
    save_default_files(directory)
    print("Saving default config file and templates.")
