#!/bin/bash
echo "Minio logs"
docker logs "$(basename $PWD)-minio-1"

echo "Mongo logs"
docker logs "$(basename $PWD)-mongo-1"

echo "API-Server logs"
docker logs "$(basename $PWD)-api-server-1"

echo "Combiner logs"
docker logs "$(basename $PWD)-combiner-1"
          
echo "Client 1 logs"
if [ "$example" == "mnist-keras" ]; then
    docker logs "$(basename $PWD)-client-1"
else
    docker logs "$(basename $PWD)-client1-1"
fi
