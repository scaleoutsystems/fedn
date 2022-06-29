#!/bin/bash
set -e

# Init venv
python -m venv .venv

# Pip deps
.venv/bin/pip install --upgrade pip
.venv/bin/pip install \
    sphinx==4.4.0 \
    sphinx_press_theme==0.8.0 \
    sphinx-autobuild==2021.3.14 \
    autopep8==1.5.7 \
    isort==5.10.1 \
    flake8==4.0.1 \
    sphinx_rtd_theme==0.5.2
.venv/bin/pip install -e fedn