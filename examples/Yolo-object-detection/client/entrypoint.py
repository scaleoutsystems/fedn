import json
import os
import fire
import numpy as np

from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

NUM_CLASSES = 10

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def _get_data_path():
    data_path = os.environ.get("FEDN_DATA_PATH", abs_path + "/data/clients/1/mnist.npz")

    return data_path

def save_darknet2fedn(darkfile, fednfile    ):

    fp = open(darkfile, "rb")
    buf = np.fromfile(fp, dtype=np.float32)
    helper.save([buf], fednfile)
    fp.close()
def save_fedn2darknet(fednfile, darkfile):

    buf = helper.load(fednfile)[0]

    with open(darkfile, "wb") as f:
        buf.tofile(f)
def init_seed(out_path="../seed.npz"):
    """Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    #weights = compile_model().get_weights()
    #helper.save(weights, out_path)
    print("hereeee")
    darkfile="yolov4-tiny.weights"
    fp = open(darkfile, "rb")
    buf = np.fromfile(fp, dtype=np.float32)
    helper.save([buf], out_path)
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
    darkfile = "example.weights"

    save_fedn2darknet(in_model_path, darkfile)
    #save_darknet2fedn("/Users/sowmyasriseenivasan/workspaces/fedn/examples/Yolo-object-detection/client/darknet/yolov4-tiny.weights","client")

    # cli call to darknet to train darkfile and save it to e.g. darkfile_upd.weights
    import subprocess

    # Paths to your files
    data_file = "darknet/obj.data"
    cfg_file = "darknet/yolov4-tiny.cfg"
    yolo_converted_weights = "example.weights"  # Pretrained weights file

    # Darknet executable path
    darknet_path = "./darknet/darknet"  # Make sure this path is correct

    # Command to train YOLO using Darknet
    command = [darknet_path, "detector", "train", data_file, cfg_file, yolo_converted_weights]

    # Run the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")

    #save_darknet2fedn("yolov4-tiny_last.weights", out_model_path)

    # saving metadata in the same way as in our example

    metadata = {
        # num_examples are mandatory
        "num_examples": 600,
        "batch_size": batch_size,
        "epochs": 1,
        "lr": 0.001,
    }
    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)
    # Save model update (mandatory)
    save_darknet2fedn("yolov4-tiny_final.weights", "output.npz")
    helper.save("output.npz", out_model_path)
if __name__ == "__main__":
    fire.Fire(
        {
            "init_seed": init_seed,
            "train": train,
            "_get_data_path": _get_data_path,  # for testing
        }
    )
#if __name__ == "__main__":
    #train("fedn/examples/Yolo-object-detection/client/yolov4-tiny.conv.29","client")
    #save_darknet2fedn("/Users/sowmyasriseenivasan/Downloads/example_project_yolo/firstmodel/carplusbike_firstdataset.weights","/Users/sowmyasriseenivasan/workspaces/fedn/examples/Yolo-object-detection/client/output/output.npz")
    #darkfile="example.weights"
    #init_seed()
    #save_fedn2darknet("/Users/sowmyasriseenivasan/workspaces/fedn/examples/Yolo-object-detection/client/output/output.npz",darkfile)
""" def validate(in_model_path, out_json_path, data_path=None):
"""     """Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str """
"""
    # Load data
    x_train, y_train = load_data(data_path)
    x_test, y_test = load_data(data_path, is_train=False)

    # Load model
    model = compile_model()
    helper = get_helper(HELPER_MODULE)
    weights = helper.load(in_model_path)
    model.set_weights(weights)

    # Evaluate
    model_score = model.evaluate(x_train, y_train)
    model_score_test = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    # JSON schema
    report = {
        "training_loss": model_score[0],
        "training_accuracy": model_score[1],
        "test_loss": model_score_test[0],
        "test_accuracy": model_score_test[1],
    }

    # Save JSON
    save_metrics(report, out_json_path)


def predict(in_model_path, out_json_path, data_path=None):
    # Using test data for inference but another dataset could be loaded
    x_test, _ = load_data(data_path, is_train=False)

    # Load model
    model = compile_model()
    helper = get_helper(HELPER_MODULE)
    weights = helper.load(in_model_path)
    model.set_weights(weights)

    # Infer
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    # Save JSON
    with open(out_json_path, "w") as fh:
        fh.write(json.dumps({"predictions": y_pred.tolist()})) """



""" if __name__ == "__main__":
    #train("fedn/examples/Yolo-object-detection/client/yolov4-tiny.conv.29","fedn/examples/Yolo-object-detection/client/")
    save_darknet2fedn("/Users/sowmyasriseenivasan/workspaces/fedn/examples/Yolo-object-detection/client/darknet/yolov4-tiny.weights","client")
 """