#!/bin/bash
python -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -e ../../fedn
venv/bin/pip install -r requirements.txt