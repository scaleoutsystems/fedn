import math
import os
import sys

import yaml

import torch
from model import load_parameters, save_parameters
from data import load_data, get_classes
from fedn.utils.helpers.helpers import save_metadata

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

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

def pre_training_settings(num_class, batch_size, train_x, train_y, num_workers=2): 

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


    class MedNISTDataset(torch.utils.data.Dataset):
        def __init__(self, image_files, labels, transforms):
            self.image_files = image_files
            self.labels = labels
            self.transforms = transforms

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, index):
            return self.transforms(self.image_files[index]), self.labels[index]


    train_ds = MedNISTDataset(train_x, train_y, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers )

    return train_loader


def train(in_model_path, out_model_path, data_path=None, client_settings_path=None):
    """Complete a model update.

    Load model paramters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated paramters
    to out_model_path (picked up by the FEDn client).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data directory.
    :type data_path: str
    :param client_settings_path: path to a local client settings file.
    :type client_settings_path: str
    """
    
    if client_settings_path is None:
        client_settings_path = os.environ.get("FEDN_CLIENT_SETTINGS_PATH", dir_path + "/client_settings.yaml")

    with open(client_settings_path, 'r') as fh: # Used by CJG for local training

        try:
            client_settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise
    batch_size = client_settings['batch_size']
    max_epochs = client_settings['local_epochs']
    num_workers = client_settings['num_workers']
    sample_size = client_settings['sample_size']
    lr = client_settings['lr']

    num_class = len(get_classes(data_path))

    # Load data
    x_train, y_train = load_data(data_path, sample_size)
    train_loader = pre_training_settings(num_class, batch_size, x_train, y_train, num_workers)

    # Load parmeters and initialize model
    model = load_parameters(in_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    loss_function = torch.nn.CrossEntropyLoss()

    # Train
    epoch_loss_values = []
    # writer = SummaryWriter()

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(sample_size) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    print(f"training completed!")

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": len(x_train),
        "batch_size": batch_size,
        "epochs": max_epochs,
        "lr": lr,
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
