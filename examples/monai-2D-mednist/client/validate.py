import os
import sys

import torch
from model import load_parameters
import yaml

import torch
from model import load_parameters, save_parameters
from data import load_data, get_classes
from fedn.utils.helpers.helpers import save_metadata

from monai.data import decollate_batch, DataLoader
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism
import numpy as np
from monai.data import decollate_batch, DataLoader

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from fedn.utils.helpers.helpers import save_metrics

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def pre_validation_settings(batch_size, train_x, train_y, val_x, val_y, num_workers=2):

    val_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

    class MedNISTDataset(torch.utils.data.Dataset):
        def __init__(self, image_files, labels, transforms):
            self.image_files = image_files
            self.labels = labels
            self.transforms = transforms

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, index):
            return self.transforms(self.image_files[index]), self.labels[index]


    val_ds = MedNISTDataset(val_x, val_y, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    return val_loader



def validate(in_model_path, out_json_path, data_path=None, client_settings_path=None):
    """Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    :param client_settings_path: The path to the local client settings file.
    :type client_settings_path: str
    """
    
    if client_settings_path is None:
        client_settings_path = os.environ.get("FEDN_CLIENT_SETTINGS_PATH", dir_path + "/client_settings.yaml")

    with open(client_settings_path, 'r') as fh: # Used by CJG for local training

        try:
            client_settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise
    
    num_workers = client_settings['num_workers']
    sample_size = client_settings['sample_size']
    batch_size  = client_settings['batch_size']

    # Load data
    x_train, y_train = load_data(data_path, sample_size)
    x_val, y_val = load_data(data_path, sample_size, is_train=False)

    val_loader = pre_validation_settings(batch_size,  x_train, y_train, x_val, y_val, num_workers)

    # Load model
    model = load_parameters(in_model_path)
    model.eval()

    y_true = []
    y_pred = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = (
                val_data[0].to(device),
                val_data[1].to(device),
            )
            pred = model(val_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(val_labels[i].item())
                y_pred.append(pred[i].item())
 
     
    print(classification_report(y_true, y_pred, digits=4))
     
    # JSON schema
    report = {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_f1_score": f1_score(y_true, y_pred, average='macro')
    }

    # Save JSON
    save_metrics(report, out_json_path)


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
