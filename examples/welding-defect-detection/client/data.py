import os
from math import floor
import opendatasets
import shutil

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def load_labels(label_dir):
    label_files = os.listdir(label_dir)
    data = []
    for label_file in label_files:
        with open(os.path.join(label_dir, label_file), "r") as file:
            lines = file.readlines()
            for line in lines:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                data.append([class_id, x_center, y_center, width, height])
    return data


def load_data(data_path, step):
    if data_path is None:
        data_env = os.environ.get("SCALEOUT_DATA_PATH")
        if data_env is None:
            data_path = f"{abs_path}/data/clients/1"
        else:
            data_path = f"{abs_path}{data_env}"
    if step == "train":
        y = os.listdir(f"{data_path}/train/labels")
        length = len(y)
    elif step == "test":
        y = os.listdir(f"{data_path}/test/labels")
        length = len(y)
    else:
        y = os.listdir(f"{data_path}/valid/labels")
        length = len(y)

    X = f"{data_path}/data.yaml"
    return X, length


def move_data_yaml(base_dir, new_path):
    old_image_path = os.path.join(base_dir, "data.yaml")
    new_image_path = os.path.join(new_path, "data.yaml")
    shutil.copy(old_image_path, new_image_path)


def splitset(dataset, parts):
    n = len(dataset)
    local_n = floor(n / parts)
    result = []
    for i in range(parts):
        result.append(dataset[i * local_n : (i + 1) * local_n])
    return result


def build_client_folder(folder, data, idx, subdir):

    os.makedirs(f"{subdir}/{folder}/images")
    os.makedirs(f"{subdir}/{folder}/labels")
    if folder=="train":
        x = "x_train"
        y = "y_train"
    elif folder=="test":
        x = "x_test"
        y = "y_test"
    else:
        x = "x_val"
        y = "y_val"

    for image in data[x][idx]:
        old_image_path = os.path.join(f"{abs_path}/welding-defect-object-detection/The Welding Defect Dataset/\
The Welding Defect Dataset/{folder}/images", image)
        new_image_path = os.path.join(f"{subdir}/{folder}/images", image)
        shutil.move(old_image_path, new_image_path)
    for label in data[y][idx]:
        old_image_path = os.path.join(f"{abs_path}/welding-defect-object-detection/The Welding Defect Dataset/\
The Welding Defect Dataset/{folder}/labels", label)
        new_image_path = os.path.join(f"{subdir}/{folder}/labels", label)
        shutil.move(old_image_path, new_image_path)

def split(out_dir="data"):
    n_splits = int(os.environ.get("SCALEOUT_NUM_DATA_SPLITS", 1))

    # Make dir
    if not os.path.exists(f"{out_dir}/clients"):
        os.makedirs(f"{out_dir}/clients")
        opendatasets.download("https://www.kaggle.com/datasets/sukmaadhiwijaya/welding-defect-object-detection")
    # Load data and convert to dict
    X_train = [f for f in os.listdir(f"{abs_path}/welding-defect-object-detection/The Welding Defect Dataset/\
The Welding Defect Dataset/train/images")]
    X_test = [f for f in os.listdir(f"{abs_path}/welding-defect-object-detection/The Welding Defect Dataset/\
The Welding Defect Dataset/test/images")]
    X_val = [f for f in os.listdir(f"{abs_path}/welding-defect-object-detection/The Welding Defect Dataset/\
The Welding Defect Dataset/valid/images")]

    y_train = [f for f in os.listdir(f"{abs_path}/welding-defect-object-detection/The Welding Defect Dataset/\
The Welding Defect Dataset/train/labels")]
    y_test = [f for f in os.listdir(f"{abs_path}/welding-defect-object-detection/The Welding Defect Dataset/\
The Welding Defect Dataset/test/labels")]
    y_val = [f for f in os.listdir(f"{abs_path}/welding-defect-object-detection/The Welding Defect Dataset/\
The Welding Defect Dataset/valid/labels")]

    data = {
        "x_train": splitset(X_train, n_splits),
        "y_train": splitset(y_train, n_splits),
        "x_test": splitset(X_test, n_splits),
        "y_test": splitset(y_test, n_splits),
        "x_val": splitset(X_val, n_splits),
        "y_val": splitset(y_val, n_splits),
    }

    # Make splits
    folders = ["train", "test", "valid"]
    for i in range(n_splits):
        subdir = f"{out_dir}/clients/{str(i+1)}"
        if not os.path.exists(subdir):
            for folder in folders:
                build_client_folder(folder, data, i, subdir)
            move_data_yaml(f"{abs_path}/welding-defect-object-detection/The Welding Defect Dataset/\
The Welding Defect Dataset", subdir)
    # Remove downloaded directory
    if os.path.exists(f"{abs_path}/welding-defect-object-detection"):
        shutil.rmtree(f"{abs_path}/welding-defect-object-detection")


if __name__ == "__main__":
    # Prepare data if not already done
    if not os.path.exists(abs_path + "/data/clients/1"):
        split()
