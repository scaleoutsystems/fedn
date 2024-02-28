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
    up -d --build mongo minio

sleep 10

docker-compose \
    -f ../../docker-compose.yaml \
    -f docker-compose.override.yaml \
    up -d --build api-server combiner client

>&2 echo "Wait for reducer to start"
".$example/bin/python" ../../.ci/tests/examples/wait_for.py reducer

>&2 echo "Wait for combiners to connect"
".$example/bin/python" ../../.ci/tests/examples/wait_for.py combiners

>&2 echo "Upload compute package"
".$example/bin/python" ../../.ci/tests/examples/api_test.py set_package --path package.tgz --helper "$helper"

>&2 echo "Upload seed"
".$example/bin/python" ../../.ci/tests/examples/api_test.py set_seed --path seed.npz

>&2 echo "Wait for clients to connect"
".$example/bin/python" ../../.ci/tests/examples/wait_for.py clients

>&2 echo "Start session"
".$example/bin/python" ../../.ci/tests/examples/api_test.py start_session --rounds 3 --helper "$helper"

>&2 echo "Checking rounds success"
".$example/bin/python" ../../.ci/tests/examples/wait_for.py rounds

>&2 echo "Test client connection with dowloaded settings"
# Get config
".$example/bin/python" ../../.ci/tests/examples/api_test.py get_client_config --output ../../client.yaml

# Redeploy clients with config
docker-compose \
    -f ../../docker-compose.yaml \
    -f docker-compose.override.yaml \
    -f ../../.ci/tests/examples/compose-client-settings.override.yaml \
    up -d

>&2 echo "Wait for clients to reconnect"
".$example/bin/python" ../../.ci/tests/examples/wait_for.py clients

>&2 echo "Test API GET requests"
".$example/bin/python" ../../.ci/tests/examples/api_test.py test_api_get_methods

popd
>&2 echo "Test completed successfully"