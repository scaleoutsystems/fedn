#!/bin/bash
>&2 echo "Run inference"
curl -k -X POST https://localhost:8090/infer

>&2 echo "Checking inference success"
".$example/bin/python" ../../.ci/tests/examples/inference_test.py