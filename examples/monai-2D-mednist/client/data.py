import os
from math import floor
import random
import PIL
import numpy as np
import torch
import torchvision
from monai.apps import download_and_extract

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


def get_data(out_dir="data"):
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
     
    data_dir =  os.path.abspath(out_dir)
    print('data_dir:', data_dir)
    if os.path.exists(data_dir):
        print('path exist.')
        if not os.path.exists(compressed_file):
            print('compressed file does not exist, downloading and extracting data.')
            download_and_extract(resource, compressed_file, data_dir, md5)
        else:
            print('files already exist.')

def get_classes(data_path):
    """Get a list of classes from the dataset
    
    :param data_path: Path to data directory.
    :type data_path: str
    """

    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH", abs_path + "/data/MedNIST")    

    class_names = sorted(x for x in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, x)))
    return(class_names)

def load_data(data_path, sample_size=None, is_train=True):
    """Load data from disk.

    :param data_path: Path to data directory.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple"""
    
    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH", abs_path + "/data/MedNIST") 
    
    class_names =   get_classes(data_path)
    num_class = len(class_names)
    
    image_files_all = [
        [os.path.join(data_path, class_names[i], x) for x in os.listdir(os.path.join(data_path, class_names[i]))]
        for i in range(num_class)
    ]

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
    #test_frac = 0.1
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
        return val_x, val_y


if __name__ == "__main__":
    # Prepare data if not already done
    if not os.path.exists(abs_path + "/data"):
        get_data()
        #load_data('./data')
