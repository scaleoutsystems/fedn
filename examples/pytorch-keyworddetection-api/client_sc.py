"""Client SC Example for PyTorch Keyword Detection API.

This module contains the implementation of the client for the federated learning
network using PyTorch for keyword detection.
"""

import argparse
import io
from pathlib import Path

from data import get_dataloaders
from model import compile_model, load_parameters, model_hyperparams, save_parameters
from settings import BATCHSIZE_VALID, DATASET_PATH, DATASET_TOTAL_SPLITS, KEYWORDS
from torch import nn
from torch.optim import Adam
from torcheval.metrics import MulticlassAccuracy
from util import construct_api_url, read_settings

from fedn.network.clients.fedn_client import ConnectToApiResult, FednClient


def main() -> None:
    """Parse arguments and start the client."""
    parser = argparse.ArgumentParser(description="Client SC Example")
    parser.add_argument("--client-yaml", type=str, required=False, help="Settings specfic for the client (default: client.yaml)")
    parser.add_argument("--dataset-split-idx", type=int, required=True, help="Setting for which dataset split this client uses")

    parser.set_defaults(client_yaml="client.yaml")
    args = parser.parse_args()

    start_client(args.client_yaml, args.dataset_split_idx)


def start_client(client_yaml: str, dataset_split_idx: int) -> None:
    """Start the client with the given configuration and dataset split index.

    Args:
        client_yaml (str): Path to the client configuration YAML file.
        dataset_split_idx (int): Index of the dataset split to use.

    """
    DATASET_SPLIT_IDX = dataset_split_idx

    cfg = load_client_config(client_yaml)
    url = construct_api_url(cfg["api_url"], cfg.get("api_port", None))

    fedn_client = FednClient(
        train_callback=lambda params, settings: on_train(params, settings, DATASET_SPLIT_IDX),
        validate_callback=lambda params: on_validate(params, DATASET_SPLIT_IDX),
        predict_callback=lambda params: on_predict(params, DATASET_SPLIT_IDX),
    )

    configure_fedn_client(fedn_client, cfg)

    result, combiner_config = fedn_client.connect_to_api(url, cfg["token"], get_controller_config(fedn_client))

    if result != ConnectToApiResult.Assigned:
        print("Failed to connect to API, exiting.")
        return

    if not fedn_client.init_grpchandler(config=combiner_config, client_name=fedn_client.client_id, token=cfg["token"]):
        return

    fedn_client.run()


def load_client_config(client_yaml: str) -> dict:
    """Load the client configuration from a YAML file.

    Args:
        client_yaml (str): Path to the client configuration YAML file.

    Returns:
        dict: The client configuration as a dictionary.

    """
    if Path(client_yaml).exists():
        cfg = read_settings(client_yaml)
    else:
        raise Exception(f"Client yaml file not found: {client_yaml}")

    if "discover_host" in cfg:
        cfg["api_url"] = cfg["discover_host"]

    return cfg


def configure_fedn_client(fedn_client: FednClient, cfg: dict) -> None:
    """Configure the FednClient with the given settings.

    Args:
        fedn_client (FednClient): The FednClient instance to configure.
        cfg (dict): The configuration dictionary containing client settings.

    """
    fedn_client.set_name(cfg["name"])
    fedn_client.set_client_id(cfg["client_id"])


def get_controller_config(fedn_client: FednClient) -> dict:
    """Get the controller configuration for the FednClient.

    Args:
        fedn_client (FednClient): The FednClient instance.

    Returns:
        dict: The controller configuration dictionary.

    """
    return {
        "name": fedn_client.name,
        "client_id": fedn_client.client_id,
        "package": "local",
        "preferred_combiner": "",
    }


def on_train(model_params, settings, dataset_split_idx) -> tuple:
    """Train the model with the given parameters and settings.

    Args:
        model_params: The model parameters.
        settings: The training settings.
        dataset_split_idx: The index of the dataset split to use.

    Returns:
        tuple: The trained model parameters and metadata.

    """
    training_metadata = {"batchsize_train": 64, "lr": 1e-3, "n_epochs": 1}

    dataloader_train, _, _ = get_dataloaders(
        DATASET_PATH, KEYWORDS, dataset_split_idx, DATASET_TOTAL_SPLITS, training_metadata["batchsize_train"], BATCHSIZE_VALID
    )

    sc_model = compile_model(**model_hyperparams(dataloader_train.dataset))
    load_parameters(sc_model, model_params)
    optimizer = Adam(sc_model.parameters(), lr=training_metadata["lr"])
    loss_fn = nn.CrossEntropyLoss()
    n_epochs = training_metadata["n_epochs"]

    for epoch in range(n_epochs):
        sc_model.train()
        for idx, (y_labels, x_spectrograms) in enumerate(dataloader_train):
            optimizer.zero_grad()
            _, logits = sc_model(x_spectrograms)

            loss = loss_fn(logits, y_labels)
            loss.backward()

            optimizer.step()

            if idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs} | Batch: {idx + 1}/{len(dataloader_train)} | Loss: {loss.item()}")

    out_model = save_parameters(sc_model, io.BytesIO())

    metadata = {"training_metadata": {"num_examples": len(dataloader_train.dataset)}}

    return out_model, metadata


def on_validate(model_params, dataset_split_idx) -> dict:
    """Validate the model with the given parameters and dataset split index.

    Args:
        model_params: The model parameters.
        dataset_split_idx: The index of the dataset split to use.

    Returns:
        dict: The validation report containing training and validation accuracy.

    """
    dataloader_train, dataloader_valid, dataloader_test = get_dataloaders(
        DATASET_PATH, KEYWORDS, dataset_split_idx, DATASET_TOTAL_SPLITS, BATCHSIZE_VALID, BATCHSIZE_VALID
    )

    n_labels = dataloader_train.dataset.n_labels

    sc_model = compile_model(**model_hyperparams(dataloader_train.dataset))
    load_parameters(sc_model, model_params)

    def evaluate(dataloader) -> float:
        sc_model.eval()
        metric = MulticlassAccuracy(num_classes=n_labels)
        for y_labels, x_spectrograms in dataloader:
            probs, _ = sc_model(x_spectrograms)

            y_pred = probs.argmax(-1)
            metric.update(y_pred, y_labels)
        return metric.compute().item()

    return {"training_acc": evaluate(dataloader_train), "validation_acc": evaluate(dataloader_valid)}


def on_predict(model_params, dataset_split_idx) -> dict:
    """Generate predictions using the model parameters and dataset split index.

    Args:
        model_params: The model parameters.
        dataset_split_idx: The index of the dataset split to use.

    Returns:
        dict: The prediction results.

    """
    dataloader_train, _, _ = get_dataloaders(DATASET_PATH, KEYWORDS, dataset_split_idx, DATASET_TOTAL_SPLITS, BATCHSIZE_VALID, BATCHSIZE_VALID)
    sc_model = compile_model(**model_hyperparams(dataloader_train.dataset))
    load_parameters(sc_model, model_params)

    return {}


if __name__ == "__main__":
    main()
