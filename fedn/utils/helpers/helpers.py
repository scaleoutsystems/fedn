import importlib
import json

HELPER_PLUGIN_PATH = "fedn.utils.helpers.plugins.{}"


def get_helper(helper_module_name):
    """Return an instance of the helper class.

    :param helper_module_name: The name of the helper plugin module.
    :type helper_module_name: str
    :return: A helper instance.
    :rtype: class: `fedn.utils.helpers.helpers.HelperBase`
    """
    helper_plugin = HELPER_PLUGIN_PATH.format(helper_module_name)
    helper = importlib.import_module(helper_plugin)
    return helper.Helper()


def save_metadata(metadata, filename):
    """Save metadata to file.

    :param metadata: The metadata to save.
    :type metadata: dict
    :param filename: The name of the file to save to.
    :type filename: str
    """
    with open(filename + "-metadata", "w") as outfile:
        json.dump(metadata, outfile)


def save_metrics(metrics, filename):
    """Save metrics to file.

    :param metrics: The metrics to save.
    :type metrics: dict
    :param filename: The name of the file to save to.
    :type filename: str
    """
    with open(filename, "w") as outfile:
        json.dump(metrics, outfile)
