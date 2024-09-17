import os
import sys
import numpy as np
import re
import subprocess
from fedn.utils.helpers.helpers import get_helper, save_metrics

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

NUM_CLASSES = 10

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
def validate(in_model_path, out_json_path, data_path=None):
    """Validate model.
    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir("../../.darknet")
    darkfile = dir_path + "/data/1/darknet.weights"
    save_fedn2darknet(in_model_path, darkfile)
    data_file = dir_path+"/data/1/obj.data"
    cfg_file = dir_path+"/data/1/yolov4-tiny.cfg"


    darknet_path = "./darknet"  # Make sure this path is correct
    value=str(0.01)
    # Command to validate YOLO using Darknet
    command = [darknet_path, "detector", "map", data_file, cfg_file, darkfile,"-iou_thresh", value]
    result = subprocess.run(command, capture_output=True, text=True, check=True)

    # Extracting metrics using regex
    output = result.stdout
    metrics = {
        "mAP": None,
        "AP": [],
        "Average IoU": None
    }

    # Refined regex patterns
    map_pattern = re.compile(r"mean average precision \(mAP@0\.01\) = ([0-9.]+)")
    ap_pattern = re.compile(r"class_id = \d+, name = [\w]+, ap = ([0-9.]+)%")
    iou_pattern = re.compile(r"average IoU = ([0-9.]+) %")
    map_match = map_pattern.search(output)
    ap_matches = ap_pattern.findall(output)
    iou_match = iou_pattern.search(output)
    if map_match:
        metrics["mAP"] = float(map_match.group(1))
    else:
        metrics["mAP"] = 0.0

    if ap_matches:
        metrics["AP"] = [float(ap) for ap in ap_matches]
    else:
        metrics["AP"]=0.0

    if iou_match:
        metrics["Average IoU"] = float(iou_match.group(1))
    else:
        metrics["Average IoU"] = 0.0

    report = {
        "Average Precision": metrics["AP"][0] ,
        "Average IOU": metrics["Average IoU"],
        "Mean Average Precision": metrics["mAP"]
    }
    save_metrics(report, out_json_path)
if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
