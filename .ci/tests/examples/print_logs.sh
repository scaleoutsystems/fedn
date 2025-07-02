#!/bin/bash
service="$1"
example="$2"
helper="$3"

if [ "$service" == "minio" ]; then
    echo "Minio logs"
    docker logs "$(basename $PWD)-minio-1"
    exit 0
fi

if [ "$service" == "mongo" ]; then
    echo "Mongo logs"
    docker logs "$(basename $PWD)-mongo-1"
    exit 0
fi

if [ "$service" == "api-server" ]; then
    echo "API-Server logs"
    docker logs "$(basename $PWD)-api-server-1"
    exit 0
fi

if [ "$service" == "combiner" ]; then
    echo "Combiner logs"
    docker logs "$(basename $PWD)-combiner-1"
    exit 0
fi

if [ "$service" == "controller" ]; then
    echo "Controller logs"
    docker logs "$(basename $PWD)-controller-1"
    exit 0
fi

if [ "$service" == "client" ]; then
    echo "Client 0 logs"
    if [ "$example" == "mnist-keras" ]; then
        docker logs "$(basename $PWD)-client-0"
    else
        docker logs "$(basename $PWD)-client0-1"
    fi
    exit 0
fi
