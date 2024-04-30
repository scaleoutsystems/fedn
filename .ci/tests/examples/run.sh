#!/bin/bash
set -e

# Parse example name
if [ "$#" -ne 2 ]; then
    >&2 echo "Wrong number of arguments (usage: run.sh <example-name> <helper>)"
    exit 1
fi
example="$1"
helper="$2"

python -m venv ".$example"
source ".$example/bin/activate"
".$example/bin/pip" install ./fedn/ fire

>&2 echo "Start FEDn"
pushd "examples/$example"

"../../.$example/bin/fedn" package create --path client
"../../.$example/bin/fedn" run build --path client

if [ "$example" == "mnist-keras" ]; then
    docker compose \
        -f ../../docker-compose.yaml \
        -f docker-compose.override.yaml \
        up -d --build --scale client=1
else
    docker compose \
        -f ../../docker-compose.yaml \
        -f docker-compose.override.yaml \
        up -d --build combiner api-server mongo minio client1   
fi

>&2 echo "Wait for reducer to start"
python ../../.ci/tests/examples/wait_for.py reducer

>&2 echo "Wait for combiners to connect"
python ../../.ci/tests/examples/wait_for.py combiners

>&2 echo "Upload compute package"
python ../../.ci/tests/examples/api_test.py set_package --path package.tgz --helper "$helper"

>&2 echo "Wait for clients to connect"
python ../../.ci/tests/examples/wait_for.py clients

>&2 echo "Upload seed"
python ../../.ci/tests/examples/api_test.py set_seed --path seed.npz

>&2 echo "Start session"
python ../../.ci/tests/examples/api_test.py start_session --rounds 3 --helper "$helper"

>&2 echo "Checking rounds success"
python ../../.ci/tests/examples/wait_for.py rounds

>&2 echo "Test client connection with dowloaded settings"
# Get config
python ../../.ci/tests/examples/api_test.py get_client_config --output ../../client.yaml

# Redeploy clients with config
docker compose \
    -f ../../docker-compose.yaml \
    -f docker-compose.override.yaml \
    -f ../../.ci/tests/examples/compose-client-settings.override.yaml \
    up -d

>&2 echo "Wait for clients to reconnect"
python ../../.ci/tests/examples/wait_for.py clients

>&2 echo "Test API GET requests"
python ../../.ci/tests/examples/api_test.py test_api_get_methods

popd
>&2 echo "Test completed successfully"
