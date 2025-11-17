import os
from math import floor

import numpy as np
import torch
import torchvision
from torchvision import transforms

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def get_data(out_dir="data"):
    # Make dir if necessary
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Only download if not already downloaded
    if not os.path.exists(f"{out_dir}/train"):
        torchvision.datasets.CIFAR10(
            root=f"{out_dir}/train", train=True, download=True)

    if not os.path.exists(f"{out_dir}/test"):
        torchvision.datasets.CIFAR10(
            root=f"{out_dir}/test", train=False, download=True)


def load_data(data_path, is_train=True):
    """ Load data from disk.

    :param data_path: Path to data file.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple
    """
    if data_path is None:
        data_path = os.environ.get(
            "SCALEOUT_DATA_PATH", abs_path+"/data/clients/1/cifar10.pt")

    data = torch.load(data_path, weights_only=True)

    if is_train:
        X = data["x_train"]
        y = data["y_train"]
    else:
        X = data["x_test"]
        y = data["y_test"]

    return X, y


def create_knn_monitoring_dataset(out_dir="data"):
    """ Creates dataset that is used to monitor the training progress via knn accuracies """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    n_splits = int(os.environ.get("SCALEOUT_NUM_DATA_SPLITS", 2))

    # Make dir
    if not os.path.exists(f"{out_dir}/clients"):
        os.mkdir(f"{out_dir}/clients")

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.247, 0.243, 0.261])

    memoryset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                             download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    testset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                           download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))

    # save monitoring datasets to all clients
    for i in range(n_splits):
        subdir = f"{out_dir}/clients/{str(i+1)}"
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        torch.save(memoryset, f"{subdir}/knn_memoryset.pt")
        torch.save(testset, f"{subdir}/knn_testset.pt")


def load_knn_monitoring_dataset(data_path, batch_size=16):
    """ Loads the KNN monitoring dataset."""
    if data_path is None:
        data_path = os.environ.get(
            "SCALEOUT_DATA_PATH", abs_path+"/data/clients/1/cifar10.pt")

    data_directory = os.path.dirname(data_path)
    memory_path = os.path.join(data_directory, "knn_memoryset.pt")
    testset_path = os.path.join(data_directory, "knn_testset.pt")

    memoryset = torch.load(memory_path)
    testset = torch.load(testset_path)

    memoryset_loader = torch.utils.data.DataLoader(
        memoryset, batch_size=batch_size, shuffle=False)
    testset_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False)
    return memoryset_loader, testset_loader


def splitset(dataset, parts):
    n = dataset.shape[0]
    local_n = floor(n/parts)
    result = []
    for i in range(parts):
        result.append(dataset[i*local_n: (i+1)*local_n])
    return result


def split(out_dir="data"):

    n_splits = int(os.environ.get("SCALEOUT_NUM_DATA_SPLITS", 2))

    # Make dir
    if not os.path.exists(f"{out_dir}/clients"):
        os.mkdir(f"{out_dir}/clients")

    train_data = torchvision.datasets.CIFAR10(
        root=f"{out_dir}/train", train=True)
    test_data = torchvision.datasets.CIFAR10(
        root=f"{out_dir}/test", train=False)

    data = {
        "x_train": splitset(train_data.data, n_splits),
        "y_train": splitset(np.array(train_data.targets), n_splits),
        "x_test": splitset(test_data.data, n_splits),
        "y_test": splitset(np.array(test_data.targets), n_splits),
    }

    # Make splits
    for i in range(n_splits):
        subdir = f"{out_dir}/clients/{str(i+1)}"
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        torch.save({
            "x_train": data["x_train"][i],
            "y_train": data["y_train"][i],
            "x_test": data["x_test"][i],
            "y_test": data["y_test"][i],
        },
            f"{subdir}/cifar10.pt")


if __name__ == "__main__":
    # Prepare data if not already done
    if not os.path.exists(abs_path+"/data/clients/1"):
        get_data()
        split()
        create_knn_monitoring_dataset()
