#!/bin/bash
set -e

# Parse example name
if [ "$#" -lt 1 ]; then
    >&2 echo "Wrong number of arguments (usage: run_infrence.sh <example-name>)"
    exit 1
fi
example="$1"

>&2 echo "Run inference"
pushd "examples/$example"
curl -k -X POST https://localhost:8090/infer

>&2 echo "Checking inference success"
".$example/bin/python" ../../.ci/tests/examples/inference_test.py

>&2 echo "Test completed successfully"
popd