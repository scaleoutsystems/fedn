# MNIST (TensorFlow/Keras version)

This is a mimimalistic TF/Keras version of the Quickstart Tutorial (PyTorch). For more detailed explaination including a Jupyter Notebook with 
examples of API usage for starting and interacting with federated experiments, refer to that tutorial.

## Prerequisites
- [Python 3.8, 3.9 or 3.10](https://www.python.org/downloads)
- [Docker](https://docs.docker.com/get-docker)
- [Docker Compose](https://docs.docker.com/compose/install)

## Running the example (pseudo-distributed)
Clone FEDn and locate into this directory.
```sh
git clone https://github.com/scaleoutsystems/fedn.git
cd fedn/examples/mnist-keras
```

### Preparing the environment, the local data, the compute package and seed model

Start by initializing a virtual enviroment with all of the required dependencies.
```sh
bin/init_venv.sh
```

Then, to get the data you can run the following script.
```sh
bin/get_data
```

The next command splits the data in 2 parts for the clients.
```sh
bin/split_data
```
> **Note**: run with `--n_splits=N` to split in *N* parts.

Create the compute package and a seed model that you will be asked to upload in the next step.
```
bin/build.sh
```
> The files location will be `package/package.tgz` and `seed.npz`.

### Deploy FEDn 
Now we are ready to deploy FEDn and two clients with `docker-compose`.

```
docker-compose -f ../../docker-compose.yaml -f docker-compose.override.yaml up  
```

> **Note**: run with `--scale client=N` to start *N* clients.

### Run federated training 
Refer to the notebook to create your own drivers for seeding the federation and running experiments.

 https://github.com/scaleoutsystems/fedn/blob/master/examples/mnist-pytorch/API_Example.ipynb 


## Clean up
You can clean up by running `docker-compose down`.
