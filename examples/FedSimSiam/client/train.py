import os
import sys

import numpy as np
import torch
from data import load_data
from model import load_parameters, save_parameters
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils import init_lrscheduler

from scaleout.utils.helpers.helpers import save_metadata

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


class SimSiamDataset(Dataset):
    def __init__(self, x, y, is_train=True):
        self.x = x
        self.y = y
        self.is_train = is_train

    def __getitem__(self, idx):
        x = self.x[idx]
        x = Image.fromarray(x.astype(np.uint8))

        y = self.y[idx]

        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.247, 0.243, 0.261])
        augmentation = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

        if self.is_train:
            transform = transforms.Compose(augmentation)

            x1 = transform(x)
            x2 = transform(x)
            return [x1, x2], y

        else:
            transform = transforms.Compose([transforms.ToTensor(), normalize])

            x = transform(x)
            return x, y

    def __len__(self):
        return len(self.x)


def train(in_model_path, out_model_path, data_path=None, batch_size=32, epochs=1, lr=0.01):
    """ Complete a model update.

    Load model paramters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated paramters
    to out_model_path (picked up by the FEDn client).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param epochs: The number of epochs to train.
    :type epochs: int
    :param lr: The learning rate to use.
    :type lr: float
    """
    # Load data
    x_train, y_train = load_data(data_path)

    # Load parmeters and initialize model
    model = load_parameters(in_model_path)

    trainset = SimSiamDataset(x_train, y_train, is_train=True)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    model.train()

    optimizer, lr_scheduler = init_lrscheduler(
        model, 500, trainloader)

    for epoch in range(epochs):
        for idx, data in enumerate(trainloader):
            images = data[0]
            optimizer.zero_grad()
            data_dict = model.forward(images[0].to(
                device, non_blocking=True), images[1].to(device, non_blocking=True))
            loss = data_dict["loss"].mean()
            print(loss)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    # Metadata needed for aggregation server side
    metadata = {
        # num_examples are mandatory
        "num_examples": len(x_train),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save model update (mandatory)
    save_parameters(model, out_model_path)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
