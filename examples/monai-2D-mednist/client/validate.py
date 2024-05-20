import os
import sys

import torch
from model import load_parameters

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


def pre_validation_settings(num_class,train_x, train_y, test_x, test_y):

    train_transforms = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        ]
    )

    val_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=num_class)])

    class MedNISTDataset(torch.utils.data.Dataset):
        def __init__(self, image_files, labels, transforms):
            self.image_files = image_files
            self.labels = labels
            self.transforms = transforms

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, index):
            return self.transforms(self.image_files[index]), self.labels[index]

    train_ds = MedNISTDataset(train_x, train_y, val_transforms)
    train_loader = DataLoader(train_ds, batch_size=30, num_workers=1)

    test_ds = MedNISTDataset(test_x, test_y, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=30, num_workers=1)

    return train_ds, train_loader, test_ds, test_loader



def validate(in_model_path, out_json_path, data_path=None):
    """Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    # Load data
    x_train, y_train, _, _ = load_data(data_path)
    x_test, y_test = load_data(data_path, is_train=False)
   
    num_class = len(get_classes(data_path))
    train_ds, train_loader, test_ds, test_loader = pre_validation_settings(num_class, x_train, y_train, x_test, y_test)

    # Load model
    model = load_parameters(in_model_path)
    model.eval()

    y_true = []
    y_pred = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = (
                test_data[0].to(device),
                test_data[1].to(device),
            )
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
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
