#!/bin/bash
set -e

# Init venv
python -m venv .venv

# Pip deps
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e fedn