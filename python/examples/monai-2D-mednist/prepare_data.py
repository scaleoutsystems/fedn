import os
import sys
import numpy as np

import yaml
from monai.apps import download_and_extract


def split_data(data_path="data/MedNIST", splits=100, validation_split=0.9):
    # create clients
    clients = {"client " + str(i): {"train": [], "validation": []} for i in range(splits)}
    print("splits: ", splits)
    for class_ in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, class_)):
            patients_in_class = [os.path.join(class_, patient) for patient in os.listdir(os.path.join(data_path, class_))]
            np.random.shuffle(patients_in_class)
            chops = np.int32(np.linspace(0, len(patients_in_class), splits + 1))
            for split in range(splits):
                p = patients_in_class[chops[split] : chops[split + 1]]

                valsplit = np.int32(len(p) * validation_split)

                clients["client " + str(split)]["train"] += p[:valsplit]
                clients["client " + str(split)]["validation"] += p[valsplit:]

                if split == 0:
                    print("len p: ", len(p))
                    print("valsplit: ", valsplit)
                    print("p[:valsplit]: ", p[:valsplit])
                    print("p[valsplit:]: ", p[valsplit:])

    with open(os.path.join(os.path.dirname(data_path), "data_splits.yaml"), "w") as file:
        yaml.dump(clients, file, default_flow_style=False)


def get_data(out_dir="data", data_splits=10):
    """Get data from the external repository.

    :param out_dir: Path to data directory. If doesn't
    :type data_dir: str
    """
    # Make dir if necessary
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
    md5 = "0bc7306e7427e00ad1c5526a6677552d"

    compressed_file = os.path.join(out_dir, "MedNIST.tar.gz")

    data_dir = os.path.abspath(out_dir)
    print("data_dir:", data_dir)
    if os.path.exists(data_dir):
        print("path exist.")
        if not os.path.exists(compressed_file):
            print("compressed file does not exist, downloading and extracting data.")
            download_and_extract(resource, compressed_file, data_dir, md5)
        else:
            print("files already exist.")

    split_data(splits=data_splits)


if __name__ == "__main__":
    # Prepare data if not already done
    get_data(data_splits=int(sys.argv[1]))
