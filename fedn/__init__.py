import glob
import os
from os.path import basename, dirname, isfile

from fedn.network.api.client import APIClient

# flake8: noqa


modules = glob.glob(dirname(__file__) + "/fedn" + "/*.py")
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")]


_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path):
    """

    :param path:
    :return:
    """
    return os.path.join(_ROOT, "data", path)
