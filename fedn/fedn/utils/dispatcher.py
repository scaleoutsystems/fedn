import os

from fedn.common.log_config import logger
from fedn.utils.process import run_process


class Dispatcher:
    """ Dispatcher class for compute packages.

    :param config: The configuration.
    :type config: dict
    :param dir: The directory to dispatch to.
    :type dir: str
    """

    def __init__(self, config, dir):
        """ Initialize the dispatcher."""
        self.config = config
        self.project_dir = dir

    def run_cmd(self, cmd_type):
        """ Run a command.

        :param cmd_type: The command type.
        :type cmd_type: str
        :return:
        """
        try:
            cmdsandargs = cmd_type.split(' ')

            cmd = [self.config['entry_points'][cmdsandargs[0]]['command']]

            # remove the first element,  that is not a file but a command
            args = cmdsandargs[1:]

            # shell (this could be a venv, TODO: parametrize)
            if os.name == "nt":
                shell = []
            else:
                shell = ['/bin/sh', '-c']

            # add the corresponding process defined in project.yaml and append arguments from invoked command
            args = shell + [' '.join(cmd + args)]
            run_process(args=args, cwd=self.project_dir)

            logger.info('Done executing {}'.format(cmd_type))
        except IndexError:
            message = "No such argument or configuration to run."
            logger.error(message)
