"""Client SC Example for PyTorch Keyword Detection API.

This module contains the implementation of the client for the federated learning
network using PyTorch for keyword detection.
"""

import argparse
from pathlib import Path

from sc_client import SCClient
from util import construct_api_url, read_settings

from fedn.network.clients.fedn_client import ConnectToApiResult, FednClient


def main() -> None:
    """Parse arguments and start the client."""
    parser = argparse.ArgumentParser(description="Client SC Example")
    parser.add_argument("--client-yaml", type=str, required=False, help="Settings specfic for the client (default: client.yaml)")
    parser.add_argument("--dataset-split-idx", type=int, required=False, help="Setting for which dataset split this client uses")

    parser.set_defaults(client_yaml="client.yaml", dataset_split_idx=0)
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

    fedn_client = SCClient(dataset_split_idx=DATASET_SPLIT_IDX)

    configure_fedn_client(fedn_client, cfg)

    result, combiner_config = fedn_client.connect_to_api(url, cfg["token"], get_client_json(fedn_client))

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


def get_client_json(fedn_client: FednClient) -> dict:
    """Get the client json for the FednClient.

    Args:
        fedn_client (FednClient): The FednClient instance.

    Returns:
        dict: The client json dictionary.

    """
    return {
        "name": fedn_client.name,
        "client_id": fedn_client.client_id,
        "package": "local",
        "preferred_combiner": "",
    }


if __name__ == "__main__":
    main()
