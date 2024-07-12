import importlib.metadata

import fedn


def get_version(pacakge):
    # Dynamically get the version of the package
    try:
        version = importlib.metadata.version("fedn")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return version


def get_package_path():
    # Get the path of the package
    return fedn.__path__[0]
