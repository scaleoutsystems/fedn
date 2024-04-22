# ASYNC CLIENTS 
This example shows how to experiment with intermittent and asynchronous client workflows.     

## Prerequisites
- [Python 3.8, 3.9 or 3.10](https://www.python.org/downloads)
- [Docker](https://docs.docker.com/get-docker)
- [Docker Compose](https://docs.docker.com/compose/install)

## Running the example (pseudo-distributed, single host)

First, make sure that FEDn is installed (we recommend using a virtual environment)

Clone FEDn
```sh
git clone https://github.com/scaleoutsystems/fedn.git
```

Install FEDn and dependencies

``
pip install fedn
```

Or from source, standing in the folder 'fedn/fedn'

```
pip install .
```

### Prepare the example environment, the compute package and seed model

Standing in the folder fedn/examples/async-clients
```
pip install -r requirements.txt
```

Create the compute package and seed model:
```
fedn package create --path client
```

```
fedn run build --path client
```

You will now have a file 'seed.npz' in the directory.

### Running a simulation

Deploy FEDn on localhost. Standing in the the FEDn root directory: 

```
docker-compose up 
```

Initialize FEDn with the compute package and seed model

```
python init_fedn.py
```

Start simulating clients
```
python run_clients.py
```

Start the experiment / training sessions: 

```
python run_experiment.py
```

Once global models start being produced, you can start analyzing results using API Client, refer to the notebook "Experiment.ipynb" for instructions. 




