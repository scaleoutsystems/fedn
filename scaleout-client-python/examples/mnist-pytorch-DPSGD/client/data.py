import os
from math import floor

import torch
import torchvision

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def get_data(out_dir="data"):
    # Make dir if necessary
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Only download if not already downloaded
    if not os.path.exists(f"{out_dir}/train"):
        torchvision.datasets.MNIST(root=f"{out_dir}/train", transform=torchvision.transforms.ToTensor, train=True, download=True)
    if not os.path.exists(f"{out_dir}/test"):
        torchvision.datasets.MNIST(root=f"{out_dir}/test", transform=torchvision.transforms.ToTensor, train=False, download=True)


def load_data(data_path, is_train=True):
    """Load data from disk.

    :param data_path: Path to data file.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple
    """
    print("data_path is None: ", data_path is None)
    if data_path is None:
        data_path = os.environ.get("SCALEOUT_DATA_PATH", abs_path + "/data/clients/1/mnist.pt")

    print("data path: ", data_path)
    data = torch.load(data_path, weights_only=True)

    if is_train:
        X = data["x_train"]
        y = data["y_train"]
    else:
        X = data["x_test"]
        y = data["y_test"]

    # Normalize
    X = X / 255

    return X, y


def splitset(dataset, parts):
    n = dataset.shape[0]
    local_n = floor(n / parts)
    result = []
    for i in range(parts):
        result.append(dataset[i * local_n : (i + 1) * local_n])
    return result


def split(out_dir="data"):
    n_splits = int(os.environ.get("SCALEOUT_NUM_DATA_SPLITS", 2))

    # Make dir
    if not os.path.exists(f"{out_dir}/clients"):
        os.mkdir(f"{out_dir}/clients")

    # Load and convert to dict
    train_data = torchvision.datasets.MNIST(root=f"{out_dir}/train", transform=torchvision.transforms.ToTensor, train=True)
    test_data = torchvision.datasets.MNIST(root=f"{out_dir}/test", transform=torchvision.transforms.ToTensor, train=False)
    data = {
        "x_train": splitset(train_data.data, n_splits),
        "y_train": splitset(train_data.targets, n_splits),
        "x_test": splitset(test_data.data, n_splits),
        "y_test": splitset(test_data.targets, n_splits),
    }

    # Make splits
    for i in range(n_splits):
        subdir = f"{out_dir}/clients/{str(i+1)}"
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        torch.save(
            {
                "x_train": data["x_train"][i],
                "y_train": data["y_train"][i],
                "x_test": data["x_test"][i],
                "y_test": data["y_test"][i],
            },
            f"{subdir}/mnist.pt",
        )


if __name__ == "__main__":
    # Prepare data if not already done
    if not os.path.exists(abs_path + "/data/clients/1"):
        get_data()
        split()
