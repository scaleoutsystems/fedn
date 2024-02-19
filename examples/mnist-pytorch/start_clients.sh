#!/bin/bash

# Check if an argument is provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <benign_client_count> <malignant_client_count>"
    exit 1
fi

# Access the first argument
benign_client_count="$1"
malignant_client_count="$2"

# Check if the provided values are integers
if ! [[ "$benign_client_count" =~ ^[0-9]+$ ]]; then
    echo "Error: The provided value for benign client count is not an integer."
    exit 1
fi

if ! [[ "$malignant_client_count" =~ ^[0-9]+$ ]]; then
    echo "Error: The provided value for malignant client count is not an integer."
    exit 1
fi

# Loop for count of clients if benign_client_count is greater than 0
if [ "$benign_client_count" -gt 0 ]; then
    for i in $(seq 1 "$benign_client_count"); do
        echo "Starting benign_client$i"
        docker run -d \
        -v $PWD/client.yaml:/app/client.yaml \
        -v $PWD/data/clients/$i:/var/data \
        -e ENTRYPOINT_OPTS="--data_path=/var/data/mnist.pt --malicious=False" \
        --network=fedn_default \
        --name benign_client$i \
        ghcr.io/scaleoutsystems/fedn/fedn:master-mnist-pytorch run client -in client.yaml --name benign_client$i
    done
fi

# Loop for count of clients if malignant_client_count is greater than 0
if [ "$malignant_client_count" -gt 0 ]; then
    for i in $(seq 1 "$malignant_client_count"); do
        client_number=$((benign_client_count + i))
        echo "Starting malignant_client$i"
        docker run -d \
        -v $PWD/client.yaml:/app/client.yaml \
        -v $PWD/data/clients/$client_number:/var/data \
        -e ENTRYPOINT_OPTS="--data_path=/var/data/mnist.pt --malicious=True" \
        --network=fedn_default \
        --name malignant_client$i \
        ghcr.io/scaleoutsystems/fedn/fedn:master-mnist-pytorch run client -in client.yaml --name malignant_client$i
    done
fi
