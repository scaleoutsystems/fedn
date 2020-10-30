import re
import logging
from fedn.utils.process import run_process

logger = logging.getLogger(__name__)


class Dispatcher:
    def __init__(self, config, dir):
        self.config = config
        self.project_dir = dir

    def run_cmd(self, cmd_type):

        try:
            cmdsandargs = cmd_type.split(' ')

            cmd = self.config['entry_points'][cmdsandargs[0]]['command'].split(' ')

            # remove the first element,  that is not a file but a command
            args = cmdsandargs[1:]

            # add the corresponding process defined in project.yaml and append arguments from invoked command
            args = cmd + args
            #print("trying to run process {} with args {}".format(cmd, args))
            run_process(args=args, cwd=self.project_dir)

            logger.info('DONE RUNNING {}'.format(cmd_type))
        except IndexError:
            message = "No such argument or configuration to run!"
            logger.error(message)
            print(message, flush=True)
