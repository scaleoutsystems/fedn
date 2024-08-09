#!/bin/bash
set -e

# Parse example name
if [ "$#" -ne 2 ]; then
    >&2 echo "Wrong number of arguments (usage: configure.sh <example-name>)"
    exit 1
fi
example="$1"

>&2 echo "Configuring $example environment"
pushd "examples/$example"

# If example equals "mnist-keras"
if [ "$example" == "mnist-keras" ]; then

    >&2 echo "Download and prepare data"
    bin/get_data
    bin/split_data
fi
popd