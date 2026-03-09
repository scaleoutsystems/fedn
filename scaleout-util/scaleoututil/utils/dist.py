import importlib.metadata

import scaleoututil


def get_version(module_name: str = "scaleoututil") -> str:
    # Dynamically get the version of the package
    try:
        version = importlib.metadata.version(module_name)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return version


def get_package_path():
    # Get the path of the package
    return scaleoututil.__path__[0]


VERSION = get_version("scaleoututil")
