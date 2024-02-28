#!/bin/bash
set -e

# Init venv
python3.9 -m venv .flower-example

# Pip deps
.flower-example/bin/pip install --upgrade pip
.flower-example/bin/pip install -e ../../fedn
.flower-example/bin/pip install -r requirements.txt
