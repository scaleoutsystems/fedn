#!/bin/bash
set -e

# Init venv
python3 -m venv .mnist-keras

# Pip deps
.mnist-keras/bin/pip install --upgrade pip
.mnist-keras/bin/pip install -e ../../fedn
.mnist-keras/bin/pip install -r requirements-macos.txt
