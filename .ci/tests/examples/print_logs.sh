#!/bin/bash
echo "Minio logs"
docker logs fedn_minio_1

echo "Mongo logs"
docker logs fedn_mongo_1

echo "Reducer logs"
docker logs fedn_reducer_1

echo "Combiner logs"
docker logs fedn_combiner_1
          
echo "Client 1 logs"
docker logs fedn_client_1

echo "Client 2 logs"
docker logs fedn_client_2