import os
import sys
import numpy as np
import subprocess
from fedn.utils.helpers.helpers import get_helper, save_metadata

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

NUM_CLASSES = 1

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)
def save_darknet2fedn(darkfile, fednfile    ):
    fp = open(darkfile, "rb")
    buf = np.fromfile(fp, dtype=np.float32)
    helper.save([buf], fednfile)
    fp.close()
def save_fedn2darknet(fednfile, darkfile):

    buf = helper.load(fednfile)[0]
    with open(darkfile, "wb") as f:
        header = np.array([0,2, 5, 0, 0],dtype=np.int32)
        header.tofile(f)
        buf.tofile(f)
def  number_of_lines(file):
    with open(file, "r") as f:
        lines = f.readlines()
        line_count=len(lines)
    return line_count
def train(in_model_path, out_model_path, data_path=None, batch_size=64, epochs=1):
    """Complete a model update.
    Load model paramters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated paramters
    to out_model_path (picked up by the FEDn client).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir("../../.darknet")
    darkfile = dir_path + "/data/1/example.weights"
    save_fedn2darknet(in_model_path, darkfile)
    data_file=dir_path + "/data/1/obj.data"
    cfg_file = dir_path+"/data/1/yolov4-tiny.cfg"
    darknet_path = "./darknet"  # Make sure this path is correct
    yolo_converted_weights = dir_path + "/data/1/example.weights"
    command = [darknet_path, "detector", "train", data_file, cfg_file, yolo_converted_weights,"-dont_show"]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
    number_of_examples=number_of_lines(dir_path+"/data/1/train.txt")
    metadata = {
        # num_examples are mandatory
        "num_examples": number_of_examples,
        "batch_size": 64,
        "epochs": 1,
        "lr": 0.01,
    }
    save_metadata(metadata, out_model_path)
    save_darknet2fedn(dir_path+"/data/1/yolov4_tiny/yolov4-tiny_final.weights", out_model_path)
if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
