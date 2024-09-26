import os
from math import floor

import numpy as np
import requests
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def get_data(out_dir="data"):
    # Generate random int between 1 and 10 for split id, set seed for reproducibility
    split_id = np.random.randint(1, 11)

    # set split id as environment variable
    os.environ["FEDN_DATA_SPLIT_ID"] = str(split_id)

    if not os.path.exists(out_dir + f"/clients/{split_id}"):
        # create directory for data
        os.makedirs(out_dir + f"/clients/{split_id}")

    # use requests to download the data from url
    url = f"https://storage.googleapis.com/public-scaleout/mnist-pytorch/data/clients/{split_id}/mnist.pt"
    # download into out_dir
    r = requests.get(url)
    if r.status_code == 200:
        with open(f"{out_dir}/clients/{split_id}/mnist.pt", "wb") as f:
            f.write(r.content)
        print(f"Downloaded data from {url}")
    else:
        print(f"Failed to download data from {url}")


def load_data(data_path, is_train=True):
    """Load data from disk.

    :param data_path: Path to data file.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple
    """
    if data_path is None:
        split_id = 0
        for id in range(1, 11):
            if os.path.exists(f"{abs_path}/data/clients/{id}/mnist.pt"):
                split_id = id
                print(f"Found data at {abs_path}/data/clients/{id}/mnist.pt")
                break
        print(f"Using split id {split_id}")
        data_path = f"{abs_path}/data/clients/{split_id}/mnist.pt"
        data_path = os.environ.get("FEDN_DATA_PATH", data_path)
        # check if data_path is a file
        if not os.path.isfile(data_path):
            print(f"Data file {data_path} not found.")
            raise FileNotFoundError

    data = torch.load(data_path)

    if is_train:
        X = data["x_train"]
        y = data["y_train"]
    else:
        X = data["x_test"]
        y = data["y_test"]

    # Normalize
    X = X / 255

    return X, y


if __name__ == "__main__":
    # Prepare data if not already done
    get_data()
