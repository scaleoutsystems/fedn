# LOAD TEST 
This example can be used as a load test for FEDn.  

No actual machine learning is being done - the clients generate a 
random array of a configurable size. In this way a developer can
test the performance / scalability of a given FEDn network in a flexible
way simply by shuffling around and aggregating numeric arrays. 

## Prerequisites
- [Python 3.8, 3.9 or 3.10](https://www.python.org/downloads)
- [Docker](https://docs.docker.com/get-docker)
- [Docker Compose](https://docs.docker.com/compose/install)

## Running the example (pseudo-distributed, single host)

Clone FEDn and locate into this directory.
```sh
git clone https://github.com/scaleoutsystems/fedn.git
cd fedn/examples/load-test
```

### Preparing the environment, the local data, the compute package and seed model

Install FEDn: 
```
pip install fedn
```

Standing in examples/load-test
```
pip install -r requirements.txt
```

Create the compute package and a seed model that you will be asked to upload in the next step.
```
tar -czvf package.tgz client
```

```
python client/entrypoint init_seed
```

### Deploy FEDn and two clients
docker-compose -f ../../docker-compose.yaml -f docker-compose.override.yaml up 

### Initialize the FEDn network 
Edit 'init_fedn.py' to configure the FEDn host (controller) to connect to, then
```
python init_fedn.py
```

Launch clients
> **Note**: run with `--scale client=N` to start *N* clients.

## Clean up
You can clean up by running `docker-compose down -v`.
