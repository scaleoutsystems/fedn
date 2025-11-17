"""The FEDn client package is responsible for executing the federated learning tasks, including ML model training and validation. It's the acting gRPC client for the federated network.
The client first connacts the centralized controller to receive :class:`scaleout.network.combiner.Combiner` assingment. The client then connects to the combiner and
sends requests to the combiner to receive model updates and send model updates.
"""
# ruff: noqa: E501

from scaleout.client.fedn_client import FednClient
from scaleoututil.api.client import APIClient

__all__ = ["FednClient", "APIClient"]
