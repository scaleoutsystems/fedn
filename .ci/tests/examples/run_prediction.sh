#!/bin/bash
set -e

# Parse example name
if [ "$#" -lt 1 ]; then
    >&2 echo "Wrong number of arguments (usage: run_infrence.sh <example-name>)"
    exit 1
fi
example="$1"

>&2 echo "Run prediction"
pushd "examples/$example"
curl -k -X POST https://localhost:8090/predict

>&2 echo "Checking prediction success"
".$example/bin/python" ../../.ci/tests/examples/prediction_test.py

>&2 echo "Test completed successfully"
popd