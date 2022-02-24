#!/bin/bash
set -e

# Init venv
python -m venv .mnist-keras

# Pip deps
.mnist-keras/bin/pip install --upgrade pip
.mnist-keras/bin/pip install -e ../../fedn
.mnist-keras/bin/pip install -r requirements.txt