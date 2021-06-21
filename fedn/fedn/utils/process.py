import subprocess
import logging

logger = logging.getLogger()

def run_process(args, cwd):
    status = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    #print(status)
    def check_io():
        while True:
            output = status.stdout.readline().decode()
            if output:
                logger.log(logging.INFO, output)
            else:
                break


    # keep checking stdout/stderr until the child exits
    while status.poll() is None:
        check_io()
