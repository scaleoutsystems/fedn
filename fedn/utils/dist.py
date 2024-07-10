import importlib.metadata
import os
from pathlib import Path

def get_version(pacakge):
    # Dynamically get the version of the package
    try:
        version = importlib.metadata.version("fedn")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return version
def get_absolute_path(env_var):
    default_path = ".././fedn/config/settings-reducer.yaml.template"
    file_path = os.environ.get(env_var, default_path)
    # Resolve the absolute path
    absolute_file_path = Path(file_path).resolve()
    return absolute_file_path
