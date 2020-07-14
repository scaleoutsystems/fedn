SHELL := /bin/bash
proto :
	pushd sdk ; ./genprot.sh; popd

up :
	docker-compose -f docker-compose.yaml -f mnist-clients.yaml -f combiner.yaml \
	-f reducer.yaml up --build --remove-orphans
