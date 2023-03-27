import importlib

PLUGIN_PATH = "fedn.utils.plugins.{}"


def get_helper(helper_module_name):
    """ Return an instance of the helper class.

    :param helper_module_name (str): The name of the helper plugin module.
    :return: A helper instance.
    """
    helper_plugin = PLUGIN_PATH.format(helper_module_name)
    helper = importlib.import_module(helper_plugin)
    return helper.Helper()
