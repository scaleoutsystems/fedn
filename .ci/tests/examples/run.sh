#!/bin/bash
set -e

# Parse example name
if [ "$#" -ne 2 ]; then
    >&2 echo "Wrong number of arguments (usage: run.sh <example-name> <helper>)"
    exit 1
fi
example="$1"
helper="$2"

>&2 echo "Configuring $example environment"
pushd "examples/$example"
bin/init_venv.sh

>&2 echo "Download and prepare data"
bin/get_data
bin/split_data

>&2 echo "Build compute package and seed"
bin/build.sh

>&2 echo "Start FEDn"
docker-compose \
    -f ../../docker-compose.yaml \
    -f docker-compose.override.yaml \
    up -d --build

>&2 echo "Wait for reducer to start"
sleep 10 # TODO: add API call to check when ready

>&2 echo "Upload compute package"
curl -k -X POST \
    -F file=@package.tgz \
    -F helper="$helper" \
    https://localhost:8090/context
printf '\n'

>&2 echo "Upload seed"
curl -k -X POST \
    -F seed=@seed.npz \
    https://localhost:8090/models
printf '\n'

>&2 echo "Wait for clients to connect"
sleep 10 # TODO: add API call to check when ready

>&2 echo "Start round"
curl -k -X POST \
    -F rounds=3 \
    -F validate=True \
    https://localhost:8090/control
printf '\n'

>&2 echo "Checking rounds success"
".$example/bin/python" ../../.ci/tests/examples/is_success.py

popd
>&2 echo "Test completed successfully"