import argparse
import io
import os
import uuid
import time

import torch
from data import get_data_loader
from init_seed import load_parameters, save_parameters
from torch import nn, optim

from config import settings
from fedn import FednClient
from fedn.network.clients.fedn_client import ConnectToApiResult
from fedn.utils.helpers.helpers import get_helper

helper = get_helper("numpyhelper")

ROUND_NO = 0

def get_api_url(api_url: str, api_port: int, secure: bool = False):
    if secure:
        url = f"https://{api_url}:{api_port}" if api_port else f"https://{api_url}"
    else:
        url = f"http://{api_url}:{api_port}" if api_port else f"http://{api_url}"
    if not url.endswith("/"):
        url += "/"
    return url


def on_train(in_model, client_settings):
    # Save model to temp file

    global ROUND_NO
    ROUND_NO += 1

    inpath = helper.get_tmp_path()
    with open(inpath, "wb") as fh:
        fh.write(in_model.getbuffer())

    # Load model from temp file
    resnet18 = load_parameters(inpath)
    os.unlink(inpath)

    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet18 = resnet18.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    weight_decay = 5e-4
    optimizer = optim.Adam(resnet18.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Get data loader for trainset
    trainloader = get_data_loader(
        num_splits=settings["N_SPLITS"],
        balanced=settings["BALANCED"],
        iid=settings["IID"],
        is_train=True,
        batch_size=settings["BATCH_SIZE"],
        split_id=1,
    )

    # Calculate number of batches
    num_batches = len(trainloader)

    # Training loop
    num_epochs = settings["EPOCHS"]
    for epoch in range(num_epochs):
        resnet18.train()

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")

        scheduler.step()

    # Save model parameters
    outpath = helper.get_tmp_path()
    save_parameters(resnet18, outpath)
    with open(outpath, "rb") as fr:
        out_model = io.BytesIO(fr.read())
    os.unlink(outpath)

    # Return model and metadata
    training_metadata = {
        "num_examples": len(trainloader.dataset),
        "batch_size": settings["BATCH_SIZE"],
        "epochs": num_epochs,
        "lr": learning_rate,
    }
    metadata = {"training_metadata": training_metadata}
    if ROUND_NO % 2 == 0:
        print(f"Simulating straggler client, sleeping for 60 seconds,{ROUND_NO}")
        time.sleep(100)  # Simulate some delay to show straggler effect
    
    return out_model, metadata


def on_validate(in_model):
    # Save model to temp file
    inpath = helper.get_tmp_path()
    with open(inpath, "wb") as fh:
        fh.write(in_model.getbuffer())

    # Load model from temp file
    resnet18 = load_parameters(inpath)
    os.unlink(inpath)

    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet18 = resnet18.to(device)
    resnet18.eval()

    criterion = nn.CrossEntropyLoss()

    # Calculate training metrics
    trainloader = get_data_loader(
        num_splits=settings["N_SPLITS"],
        balanced=settings["BALANCED"],
        iid=settings["IID"],
        is_train=True,
        batch_size=settings["BATCH_SIZE"],
        split_id=1,
    )
    train_loss = 0
    train_correct = 0
    train_total = 0

    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

    train_accuracy = train_correct / train_total
    train_loss = train_loss / len(trainloader)

    # Calculate test metrics
    testloader = get_data_loader(
        is_train=False,
        batch_size=settings["BATCH_SIZE"],
    )
    test_loss = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_accuracy = test_correct / test_total
    test_loss = test_loss / len(testloader)

    metrics = {
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "train_loss": train_loss,
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR100 Client")
    parser.add_argument("--split-id", type=int, required=True, help="The split ID")
    args = parser.parse_args()

    client = FednClient(train_callback=on_train, validate_callback=on_validate)
    url = get_api_url(settings["DISCOVER_HOST"], settings["DISCOVER_PORT"], settings["SECURE"])
    client.set_name(f"cifar100-client-{args.split_id}")
    client.set_client_id(str(uuid.uuid4()))

    controller_config = {
        "name": client.name,
        "client_id": client.client_id,
        "package": "local",
        "preferred_combiner": "",
    }

    result, combiner_config = client.connect_to_api(url=url, token=settings["CLIENT_TOKEN"], json=controller_config)

    if result != ConnectToApiResult.Assigned:
        print("Failed to connect to API, exiting.")
        exit(1)

    print("Connected to API, got combiner config: {}".format(combiner_config))

    combiner_config.host = "localhost"
    result = client.init_grpchandler(config=combiner_config, client_name=client.client_id, token=settings["CLIENT_TOKEN"])

    if not result:
        print("Failed to initialize gRPC handler, exiting.")
        exit(1)

    client.run()
