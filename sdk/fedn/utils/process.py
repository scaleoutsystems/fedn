import subprocess


def run_process(args, cwd):
    status = subprocess.run(args, cwd=cwd)
