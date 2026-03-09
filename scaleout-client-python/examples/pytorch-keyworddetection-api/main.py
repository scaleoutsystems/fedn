"""Client SC Example for PyTorch Keyword Detection API.

This module contains the implementation of the client for the federated learning
network using PyTorch for keyword detection.
"""

import argparse
from pathlib import Path

from sc_client import SCClient
from util import construct_api_url, read_settings

from scaleout.client.edge_client import ConnectToApiResult, EdgeClient


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

    sc_client = SCClient(dataset_split_idx=DATASET_SPLIT_IDX)

    configure_edge_client(sc_client, cfg)

    result, combiner_config = sc_client.connect_to_api(url, json=get_client_json(sc_client), token=cfg["token"])

    if result != ConnectToApiResult.Assigned:
        print(f"Failed to connect to API with result: {combiner_config}, exiting.")
        return

    if not sc_client.init_grpchandler(config=combiner_config, token=cfg["token"]):
        return
    
    print("Client successfully connected to API and initialized gRPC handler, starting client run loop.")
    sc_client.run()


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


def configure_edge_client(edge_client: EdgeClient, cfg: dict) -> None:
    """Configure the EdgeClient with the given settings.

    Args:
        edge_client (EdgeClient): The EdgeClient instance to configure.
        cfg (dict): The configuration dictionary containing client settings.

    """
    edge_client.set_name(cfg["name"])
    edge_client.set_client_id(cfg["client_id"])


def get_client_json(edge_client: EdgeClient) -> dict:
    """Get the client json for the EdgeClient.

    Args:
        edge_client (EdgeClient): The EdgeClient instance.

    Returns:
        dict: The client json dictionary.

    """
    return {
        "name": edge_client.name,
        "client_id": edge_client.client_id,
        "package": "local",
        "preferred_combiner": "",
    }


if __name__ == "__main__":
    main()
