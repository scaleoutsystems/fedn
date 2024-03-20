# ASYNC SIMULATION 
This example is intended as a tool to experiment with asynchronous client workflows.     

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

``
pip install fedn
```

Or from source, standing in the folder 'fedn/fedn'

```
pip install .
```

Standing in examples/async-simulation
```
pip install -r requirements.txt
```

Create the compute package and seed model:
```
tar -czvf package.tgz client
```

```
python client/entrypoint init_seed
```

### Running a simulation

Deploy FEDn on localhost. From the FEDn root directory: 

```
docker-compose up 
```

Initialize FEDn with the compute package and seed model

```
python init_fedn.py
```

