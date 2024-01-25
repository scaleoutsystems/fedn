#!/bin/bash
echo "Generating protocol"
python3 -m grpc_tools.protoc -I=. --python_out=. --grpc_python_out=. fedn/network/grpc/*.proto
echo "DONE"
