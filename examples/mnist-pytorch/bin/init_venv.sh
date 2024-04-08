#!/bin/bash
set -e

# Init venv
python -m venv .mnist-pytorch

# Pip deps
.mnist-pytorch/bin/pip install --upgrade pip
.mnist-pytorch/bin/pip install -e ../../fedn
.mnist-pytorch/bin/pip install -r requirements.txt