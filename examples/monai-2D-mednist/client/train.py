import os
import sys

import numpy as np
import torch
import yaml
from data import MedNISTDataset
from model import load_parameters, save_parameters
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)

from scaleout.utils.helpers.helpers import save_metadata

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

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
        client_settings_path = os.environ.get("SCALEOUT_CLIENT_SETTINGS_PATH", dir_path + "/client_settings.yaml")

    with open(client_settings_path, "r") as fh:  # Used by CJG for local training
        try:
            client_settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError:
            raise

    batch_size = client_settings["batch_size"]
    max_epochs = client_settings["local_epochs"]
    num_workers = client_settings["num_workers"]
    split_index = os.environ.get("SCALEOUT_DATA_SPLIT_INDEX")
    lr = client_settings["lr"]

    if data_path is None:
        data_path = os.environ.get("SCALEOUT_DATA_PATH")

    with open(os.path.join(os.path.dirname(data_path), "data_splits.yaml"), "r") as file:
        clients = yaml.safe_load(file)

    image_list = clients["client " + str(split_index)]["train"]
    train_ds = MedNISTDataset(data_path=data_path+"/MedNIST/", transforms=train_transforms, image_files=image_list)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

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
            print(f"{step}/{len(train_loader)}, " f"train_loss: {loss.item():.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    print("training completed!")

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": len(train_loader),
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
