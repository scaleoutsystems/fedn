#!/bin/bash
set -e

# Init seed
client/entrypoint init_seed

# Make compute package
tar -czvf package.tgz client