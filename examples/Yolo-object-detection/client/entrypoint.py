import json
import os
import fire
import numpy as np
import subprocess
from fedn.utils.dist import get_package_path
from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

NUM_CLASSES = 10

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def _get_data_path():
    data_path = os.environ.get("FEDN_DATA_PATH")
    return data_path

def save_darknet2fedn(darkfile, fednfile    ):

    fp = open(darkfile, "rb")
    header=np.fromfile(fp,dtype=np.int32,count=5)
    print(header)
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
def init_seed(out_path="../seed.npz"):
    """Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    print(get_package_path())
    darkfile="data/yolov4-tiny.weights"
    fp = open(darkfile, "rb")
    buf = np.fromfile(fp, dtype=np.float32)
    helper.save([buf], out_path)
    print("saveddddd")
    fp.close()

def train(in_model_path, out_model_path, data_path=None, batch_size=32, epochs=1):
    """Complete a model update.
    Load model paramters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated paramters
    to out_model_path (picked up by the FEDn client).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param epochs: The number of epochs to train.
    :type epochs: int
    """
    os.chdir("darknet")
    data_path = _get_data_path()
    darkfile = data_path + "/example.weights"
    save_fedn2darknet(in_model_path, darkfile)

    data_file=data_path + "/obj.data"
    print("hereee",data_path)
    cfg_file = data_path+"/yolov4-tiny.cfg"
    # Pretrained weights file
    # Darknet executable path
    darknet_path = "./darknet"  # Make sure this path is correct
    # Command to train YOLO using Darknet
    yolo_converted_weights = "example.weights"
    command = [darknet_path, "detector", "train", data_file, cfg_file, yolo_converted_weights,"-dont_show"]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
    number_of_examples=number_of_lines(data_path+"/train.txt")
    print(number_of_examples)
    metadata = {
        # num_examples are mandatory
        "num_examples": number_of_examples,
        "batch_size": 64,
        "epochs": 1,
        "lr": 0.00261,
    }
    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)
    # Save model update (mandatory)
    print(out_model_path)
    save_darknet2fedn(data_path+"/yolov4_tiny/yolov4-tiny_final.weights", out_model_path)
    #helper.save("output.npz", out_model_path)


def validate(in_model_path, out_json_path, data_path=None):
    """Validate model.
    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
     """

    os.chdir("darknet")
    #save model to darkfile
    data_path = _get_data_path()
    darkfile = data_path + "/darknet.weights"
    save_fedn2darknet(in_model_path, darkfile)
    data_file = data_path+"/obj.data"
    cfg_file = data_path+"/yolov4-tiny.cfg"


    darknet_path = "./darknet"  # Make sure this path is correct

    #darkfile = "/Users/sowmyasriseenivasan/yolov4-tiny_last.weights"
    value=str(0.01)
    # Command to validate YOLO using Darknet
    command = [darknet_path, "detector", "map", data_file, cfg_file, darkfile,"-iou_thresh", value]
    print("getcwd: ", os.getcwd())
    # Run the command
    try:
        output_line = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"cd drror during training: {e}")

    #print("output:")
    #print(output_line.stdout)
    #print("-------------------------------------------------------")
    result_data = {}
    for ln, line in enumerate(output_line.stdout.split("\n")):

        #print("-- ", ln)
        for line_ in line.split(", "):
            if len(line_.split(" = "))>1:
                result_data[line_.split(" = ")[0]] = line_.split(" = ")[1]
    print(result_data)

    #print(result_data['ap'])
    #print(result_data['ap'][:-2])

    #print("ap: ", float(result_data['ap'].split("%")[0]))
    #print("average IoU: ", float(result_data['average IoU'].split("%")[0]))

    #print(" mean average precision (mAP@0.50)", float(result_data[' mean average precision (mAP@0.50)'].split("%")[0]))

        # Metadata needed for aggregation server side
    # JSON schema
    report = {
        "Average Precision": float(result_data['ap'].split("%")[0]),
        "Average IOU": float(result_data['average IoU'].split("%")[0]),
        "Mean Average Precision": float(result_data[' mean average precision (mAP@0.50)'].split("%")[0])
    }
    print(report)
    # Save JSON
    save_metrics(report, out_json_path)

if __name__ == "__main__":
    fire.Fire(
        {
            "init_seed": init_seed,
            "_get_data_path": _get_data_path,
            "train": train,
            "validate": validate# for testing

        }
    )
    #train("/home/sowmya/fedn/examples/Yolo-object-detection/package/client/output.npz","/tmp/tmpc4ibx4py")

