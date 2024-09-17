import os
import numpy as np
import subprocess

from fedn.utils.helpers.helpers import get_helper

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)
dir_path = os.path.dirname(os.path.realpath(__file__))

def git_clone(repo_url="https://github.com/AlexeyAB/darknet.git", clone_dir="../.darknet"):
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../.darknet"))
    if clone_dir:
        target_path = clone_dir
    else:
        target_path = parent_dir
    command = ["git", "clone", repo_url,target_path]

    try:
        subprocess.run(command, check=True)
        print(f"Successfully cloned {repo_url}")
    except subprocess.CalledProcessError as e:
        print(f"Error during cloning: {e}")
def init_seed(out_path="../seed.npz"):
    """Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    darkfile="../yolov4-tiny.weights"
    fp = open(darkfile, "rb")
    buf = np.fromfile(fp, dtype=np.float32)
    helper.save([buf], out_path)
    fp.close()
if __name__ == "__main__":
    init_seed("../seed.npz")
    git_clone()


