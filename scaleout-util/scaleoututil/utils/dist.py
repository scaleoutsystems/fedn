import importlib.metadata

import scaleoututil


def get_version():
    # Dynamically get the version of the package
    try:
        version = importlib.metadata.version("scaleoututil")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return version


def get_package_path():
    # Get the path of the package
    return scaleoututil.__path__[0]
