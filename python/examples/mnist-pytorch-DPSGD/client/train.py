import os
import sys

import numpy as np
import torch
import yaml
from data import load_data
from model import load_parameters, save_parameters
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from scaleout.utils.helpers.helpers import save_metadata

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


# Define a custom Dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_data = self.x_data[idx]
        y_data = self.y_data[idx]
        return x_data, y_data


MAX_PHYSICAL_BATCH_SIZE = 32
EPOCHS = 1
EPSILON = 1000.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.2
GLOBAL_ROUNDS = 10
HARDLIMIT = True


def train(in_model_path, out_model_path, data_path=None, batch_size=32, lr=0.01):
    """Complete a model update.

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
    with open("../../client_settings.yaml", "r") as fh:
        try:
            settings = yaml.safe_load(fh)
            EPSILON = float(settings["epsilon"])
            DELTA = float(settings["delta"])
            MAX_GRAD_NORM = float(settings["max_grad_norm"])
            GLOBAL_ROUNDS = int(settings["global_rounds"])
            HARDLIMIT = bool(settings["hardlimit"])
            global MAX_PHYSICAL_BATCH_SIZE
            MAX_PHYSICAL_BATCH_SIZE = int(settings["max_physical_batch_size"])
        except yaml.YAMLError as exc:
            print(exc)

    # Load data
    x_train, y_train = load_data(data_path)

    # Load parmeters and initialize model
    model = load_parameters(in_model_path)

    # Train
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    privacy_engine = PrivacyEngine()

    if os.path.isfile("privacy_accountant.state"):
        privacy_engine.accountant = torch.load("privacy_accountant.state")

    trainset = CustomDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2)

    try:
        epsilon_spent = privacy_engine.get_epsilon(DELTA)
    except ValueError:
        epsilon_spent = 0
    print("epsilon before training: ", epsilon_spent)

    round_epsilon = np.sqrt((epsilon_spent / EPSILON * np.sqrt(GLOBAL_ROUNDS)) ** 2 + 1) * EPSILON / np.sqrt(GLOBAL_ROUNDS)

    print("target epsilon: ", round_epsilon)
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=EPOCHS,
        target_epsilon=round_epsilon,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dp(model, train_loader, optimizer, EPOCHS, device, privacy_engine)
    try:
        print("epsilon after training: ", privacy_engine.get_epsilon(DELTA))
    except ValueError:
        print("cant calculate epsilon")

    if HARDLIMIT and privacy_engine.get_epsilon(DELTA) < EPSILON:
        # Metadata needed for aggregation server side
        metadata = {
            # num_examples are mandatory
            "num_examples": len(x_train),
            "batch_size": batch_size,
            "epochs": EPOCHS,
            "lr": lr,
        }

        # Save JSON metadata file (mandatory)
        save_metadata(metadata, out_model_path)

        # Save model update (mandatory)
        save_parameters(model, out_model_path)
    else:
        print("Epsilon too high, not saving model")

    # Save privacy accountant
    torch.save(privacy_engine.accountant, "privacy_accountant.state")


def train_dp(model, train_loader, optimizer, epoch, device, privacy_engine):
    model.train()
    criterion = torch.nn.NLLLoss()  # nn.CrossEntropyLoss()
    with BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, optimizer=optimizer) as memory_safe_data_loader:
        for i, (images, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
