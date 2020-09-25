#!/bin/bash
echo "Generating protocol"
python3 -m grpc_tools.protoc -I=. --python_out=. --grpc_python_out=. fedn/common/net/grpc/*.proto
echo "DONE"
