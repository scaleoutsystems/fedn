import os
import sys

import torch
from model import load_parameters, save_parameters

from data import load_data
from fedn.utils.helpers.helpers import save_metadata

from opacus import PrivacyEngine
from torch.utils.data import Dataset

import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
# Define a custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_data = self.x_data[idx]
        y_data = self.y_data[idx]
        return x_data, y_data


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

MAX_GRAD_NORM = 1.2
FINAL_EPSILON = 8.0
GLOBAL_ROUNDS = 4
EPOCHS = 5
EPSILON = FINAL_EPSILON/GLOBAL_ROUNDS
DELTA = 1e-5
HARDLIMIT = False

MAX_PHYSICAL_BATCH_SIZE = 32

def train(in_model_path, out_model_path, data_path=None, batch_size=32, epochs=1, lr=0.01):
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
    # Load data
    print("data_path: ", data_path)
    x_train, y_train = load_data(data_path)
    trainset = CustomDataset(x_train, y_train)
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    # Load parmeters and initialize model
    model = load_parameters(in_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    # Load epsilon
    if os.path.isfile("epsilon.npy"):

        tot_epsilon = np.load("epsilon.npy")
        print("load consumed epsilon: ", tot_epsilon)

    else:

        print("initiate tot_epsilon")
        tot_epsilon = 0.

    # Train
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=EPOCHS,
        target_epsilon=EPSILON,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )

    print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")



    for epoch in range(EPOCHS):
        train_dp(model, train_loader, optimizer, epoch + 1, device, privacy_engine)

    d_epsilon = privacy_engine.get_epsilon(DELTA)
    print("epsilon spent: ", d_epsilon)
    tot_epsilon = np.sqrt(tot_epsilon**2 + d_epsilon**2)
    print("saving tot_epsilon: ", tot_epsilon)
    np.save("epsilon.npy", tot_epsilon)

    if HARDLIMIT and tot_epsilon >= FINAL_EPSILON:
        print("DP Budget Exceeded: The differential privacy budget has been exhausted, no model updates will be applied to preserve privacy guarantees.")

    else:
        # Metadata needed for aggregation server side
        metadata = {
            # num_examples are mandatory
            "num_examples": len(x_train),
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
        }

        # Save JSON metadata file (mandatory)
        save_metadata(metadata, out_model_path)

        # Save model update (mandatory)
        save_parameters(model, out_model_path)

def accuracy(preds, labels):
    return (preds == labels).mean()





def train_dp(model, train_loader, optimizer, epoch, device, privacy_engine):
    model.train()
    criterion = torch.nn.NLLLoss()  # nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (i + 1) % 200 == 0:
                epsilon = privacy_engine.get_epsilon(DELTA)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
                )

if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
