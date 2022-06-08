#!/bin/bash
echo "Minio logs"
docker logs ${PWD}_minio_1

echo "Mongo logs"
docker logs ${PWD}_mongo_1

echo "Reducer logs"
docker logs ${PWD}_reducer_1

echo "Combiner logs"
docker logs ${PWD}_combiner_1
          
echo "Client 1 logs"
docker logs ${PWD}_client_1

echo "Client 2 logs"
docker logs ${PWD}_client_2