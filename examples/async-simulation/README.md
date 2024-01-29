# ASYNC SIMULATION 
This example is intended as a test for asynchronous clients.     

## Prerequisites
- [Python 3.8, 3.9 or 3.10](https://www.python.org/downloads)
- [Docker](https://docs.docker.com/get-docker)
- [Docker Compose](https://docs.docker.com/compose/install)

## Running the example (pseudo-distributed, single host)

Clone FEDn and locate into this directory.
```sh
git clone https://github.com/scaleoutsystems/fedn.git
cd fedn/examples/async-simulation
```

### Preparing the environment, the local data, the compute package and seed model

Install FEDn and dependencies (we recommend using a virtual environment):

Standing in the folder 'fedn/fedn'

```
pip install -e .
```

From examples/async-simulation
```
pip install -r requirements.txt
```

Create the compute package and a seed model that you will be asked to upload in the next step.
```
tar -cvzf package.tgz
```

```
python client/entrypoint init_seed
```

### Deploy FEDn and two clients
docker-compose -f ../../docker-compose.yaml -f docker-compose.override.yaml up 

### Initialize the federated model 
See 'Experiments.pynb' or 'launch_client.py' to set the package and seed model.

> **Note**: run with `--scale client=N` to start *N* clients.

### Run federated training 
See 'Experiment.ipynb'. 

## Clean up
You can clean up by running `docker-compose down`.
