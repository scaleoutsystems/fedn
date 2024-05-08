""" The utils package is responsible for providing utility functions for the FEDn framework. Such as logging, checksums and other model helper functions to aggregate models.
THe helper functions is there to support aggregating various models from different ML frameworks, such as Tensorflow, PyTorch and Keras."""
# flake8: noqa
from sys import version_info

PYTHON_VERSION = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
