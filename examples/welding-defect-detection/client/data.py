import os
from math import floor
import torch
import yaml
import opendatasets
#from sklearn import preprocessing
dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def get_data(out_dir=None):
    
    # Only download if not already downloaded
    if not os.path.exists(f"{out_dir}/welding-defect-object-detection"):
        opendatasets.download('https://www.kaggle.com/datasets/sukmaadhiwijaya/welding-defect-object-detection')


def load_labels(label_dir):
    label_files = os.listdir(label_dir)
    data = []
    for label_file in label_files:
        with open(os.path.join(label_dir, label_file), 'r') as file:
            lines = file.readlines()
            for line in lines:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                data.append([class_id, x_center, y_center, width, height])
    return data


def load_data(data_path=None, is_train=True, as_yaml=True):
    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH", abs_path + "welding-defect-object-detection/The Welding Defect Dataset/The Welding Defect Dataset")

    yaml = data_path + '/data.yaml'
    path = None
    if is_train:
        path = data_path + "/train/images"
    else:
        path = data_path + "/test/images"    
    dir_list = os.listdir(path)

    if as_yaml:
        return yaml, len(dir_list)
    else:
        return dir_list, len(dir_list)


def splitset(dataset, parts):

    n = dataset.shape[0]
    local_n = floor(n / parts)
    result = []
    for i in range(parts):
        result.append(dataset[i * local_n : (i + 1) * local_n])
    return result



def split(out_dir="package"):
    n_splits = int(os.environ.get("FEDN_NUM_DATA_SPLITS", 1))

    # Make dir
    if not os.path.exists(f"{out_dir}/client"):
        os.mkdir(f"{out_dir}/client")

    # Load and convert to dict
    X_train = load_data(is_train=True, as_yaml=False)
    X_test = load_data(is_train=False, as_yaml=False)

    y_train = load_labels(abs_path + "welding-defect-object-detection/The Welding Defect Dataset/The Welding Defect Dataset/train/labels")
    y_test = load_labels(abs_path + "welding-defect-object-detection/The Welding Defect Dataset/The Welding Defect Dataset/test/labels")
    
    data = {
        "x_train": splitset(X_train, n_splits),
        "y_train": splitset(y_train, n_splits),
        "x_test": splitset(X_test, n_splits),
        "y_test": splitset(y_test, n_splits),
    }

    # Make splits
    for i in range(n_splits):
        subdir = f"{out_dir}/client/{str(i+1)}"
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        torch.save(
            {
                "x_train": data["x_train"][i],
                "y_train": data["y_train"][i],
                "x_test": data["x_test"][i],
                "y_test": data["y_test"][i],
            },
            f"{subdir}/welding.pt",
        )


if __name__ == "__main__":

    get_data()
    split()
