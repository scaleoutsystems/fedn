#!/bin/bash
echo "Minio logs"
docker logs "$(basename $PWD)_minio_1"

echo "Mongo logs"
docker logs "$(basename $PWD)_mongo_1"

echo "Dashboard logs"
docker logs "$(basename $PWD)_dashboard_1"

echo "API-Server logs"
docker logs "$(basename $PWD)_api-server_1"

echo "Combiner logs"
docker logs "$(basename $PWD)_combiner_1"
          
echo "Client 1 logs"
docker logs "$(basename $PWD)_client_1"

echo "Client 2 logs"
docker logs "$(basename $PWD)_client_2"