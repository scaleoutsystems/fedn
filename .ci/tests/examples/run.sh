#!/bin/bash
set -e

# Parse example name
if [ "$#" -ne 2 ]; then
    >&2 echo "Wrong number of arguments (usage: run.sh <example-name> <helper>)"
    exit 1
fi
example="$1"
helper="$2"

>&2 echo "Start FEDn"
pushd "examples/$example"
docker-compose \
    -f ../../docker-compose.yaml \
    -f docker-compose.override.yaml \
    up -d --build

>&2 echo "Wait for reducer to start"
".$example/bin/python" ../../.ci/tests/examples/wait_for.py reducer

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
".$example/bin/python" ../../.ci/tests/examples/wait_for.py clients

>&2 echo "Start round"
curl -k -X POST \
    -F rounds=3 \
    -F validate=True \
    https://localhost:8090/control
printf '\n'

>&2 echo "Checking rounds success"
".$example/bin/python" ../../.ci/tests/examples/wait_for.py rounds

>&2 echo "Test client connection with dowloaded settings"
# Get config
curl -k https://localhost:8090/config/download > ../../client.yaml

# Redeploy clients with config
docker-compose \
    -f ../../docker-compose.yaml \
    -f docker-compose.override.yaml \
    -f ../../.ci/tests/examples/compose-client-settings.override.yaml \
    up -d

>&2 echo "Wait for clients to reconnect"
".$example/bin/python" ../../.ci/tests/examples/wait_for.py clients

>&2 echo "Test client connection with dowloaded settings"
# Get config
curl -k https://localhost:8090/config/download > ../../client.yaml

# Redeploy clients with config
docker-compose \
    -f ../../docker-compose.yaml \
    -f docker-compose.override.yaml \
    -f ../../.ci/tests/examples/compose-client-settings.override.yaml \
    up -d

>&2 echo "Wait for clients to reconnect"
".$example/bin/python" ../../.ci/tests/examples/wait_for_clients.py

popd
>&2 echo "Test completed successfully"