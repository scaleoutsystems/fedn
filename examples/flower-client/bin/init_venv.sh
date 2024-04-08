#!/bin/bash
set -e

# Init venv
python3 -m venv .flower-client

# Pip deps
.flower-client/bin/pip install --upgrade pip
.flower-client/bin/pip install -e ../../fedn
.flower-client/bin/pip install -r requirements.txt
