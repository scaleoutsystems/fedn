import os
import random
import numpy as np
import PIL
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)

DATA_CLASSES = {"AbdomenCT": 0, "BreastMRI": 1, "CXR": 2, "ChestCT": 3, "Hand": 4, "HeadCT": 5}


def get_classes(data_path):
    """Get a list of classes from the dataset

    :param data_path: Path to data directory.
    :type data_path: str
    """
    if data_path is None:
        data_path = os.environ.get("SCALEOUT_DATA_PATH", abs_path + "/data/MedNIST")

    class_names = sorted(x for x in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, x)))
    return class_names


def load_data(data_path, sample_size=None, is_train=True):
    """Load data from disk.

    :param data_path: Path to data directory.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple
    """
    if data_path is None:
        data_path = os.environ.get("SCALEOUT_DATA_PATH", abs_path + "/data/MedNIST")

    class_names = get_classes(data_path)
    num_class = len(class_names)

    image_files_all = [[os.path.join(data_path, class_names[i], x) for x in os.listdir(os.path.join(data_path, class_names[i]))] for i in range(num_class)]

    # To make the dataset small, we are using sample_size=100 images of each class.
    if sample_size is None:
        image_files = image_files_all

    else:
        image_files = [random.sample(inner_list, sample_size) for inner_list in image_files_all]

    num_each = [len(image_files[i]) for i in range(num_class)]
    image_files_list = []
    image_class = []
    for i in range(num_class):
        image_files_list.extend(image_files[i])
        image_class.extend([i] * num_each[i])
    num_total = len(image_class)
    image_width, image_height = PIL.Image.open(image_files_list[0]).size

    print(f"Total image count: {num_total}")
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Label names: {class_names}")
    print(f"Label counts: {num_each}")

    val_frac = 0.1
    length = len(image_files_list)
    indices = np.arange(length)
    np.random.shuffle(indices)

    val_split = int(val_frac * length)
    val_indices = indices[:val_split]
    train_indices = indices[val_split:]

    train_x = [image_files_list[i] for i in train_indices]
    train_y = [image_class[i] for i in train_indices]
    val_x = [image_files_list[i] for i in val_indices]
    val_y = [image_class[i] for i in val_indices]

    print(f"Training count: {len(train_x)}, Validation count: " f"{len(val_x)}")

    if is_train:
        return train_x, train_y
    else:
        return val_x, val_y, class_names


class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, image_files, transforms):
        self.data_path = data_path
        self.image_files = image_files
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return (self.transforms(os.path.join(self.data_path, self.image_files[index])), DATA_CLASSES[os.path.dirname(self.image_files[index])])




