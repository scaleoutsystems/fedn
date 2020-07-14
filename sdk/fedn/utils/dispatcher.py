import re
import logging
from fedn.utils.process import run_process

logger = logging.getLogger(__name__)


class Dispatcher:
    def __init__(self, project):
        self.config = project.config
        self.project_dir = project.project_dir

    def run_cmd(self, cmd_type):

        cmdsandargs = cmd_type.split(' ')

        cmd = self.config['Alliance']['Member']['entry_points'][cmdsandargs[0]]['command'].split(' ')

        # remove the first element,  that is not a file but a command
        args = cmdsandargs[1:]

        # add the corresponding process defined in project.yaml and append arguments from invoked command
        args = cmd + args
        print("trying to run process {} with args {}".format(cmd, args))
        run_process(args=args, cwd=self.project_dir)

        logger.info('DONE RUNNING {}'.format(cmd_type))


def get_dispatcher(project):
    return Dispatcher(project)
