import subprocess

from fedn.common.log_config import logger


def run_process(args, cwd):
    """ Run a process and log the output.

    :param args: The arguments to the process.
    :type args: list
    :param cwd: The current working directory.
    :type cwd: str
    :return:
    """
    status = subprocess.Popen(
        args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # print(status)
    def check_io():
        """ Check stdout/stderr of the child process.

        :return:
        """
        while True:
            output = status.stdout.readline().decode()
            if output:
                logger.info(output)
            else:
                break

    # keep checking stdout/stderr until the child exits
    while status.poll() is None:
        check_io()
