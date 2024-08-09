""" The FEDn client package is responsible for executing the federated learning tasks, including ML model training and validation. It's the acting gRPC client for the federated network.
The client first connacts the centralized controller to receive :class:`fedn.network.combiner.Combiner` assingment. The client then connects to the combiner and
sends requests to the combiner to receive model updates and send model updates."""
# flake8: noqa
