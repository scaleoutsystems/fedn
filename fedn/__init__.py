import glob
import os
from os.path import basename, dirname, isfile

from fedn.network.api.client import APIClient
from fedn.network.clients.fedn_client import FednClient

# ruff: noqa: F401


modules = glob.glob(dirname(__file__) + "/fedn" + "/*.py")
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")]


_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path):
    """get_data
    :param path:
    :return:
    """
    return os.path.join(_ROOT, "data", path)
