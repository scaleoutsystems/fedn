#!/bin/bash
set -e

# Parse example name
if [ "$#" -ne 2 ]; then
    >&2 echo "Wrong number of arguments (usage: run.sh <example-name>)"
    exit 1
fi
example="$1"

>&2 echo "Configuring $example environment"
pushd "examples/$example"
bin/init_venv.sh

>&2 echo "Download and prepare data"
bin/get_data
bin/split_data

>&2 echo "Build compute package and seed"
bin/build.sh

popd