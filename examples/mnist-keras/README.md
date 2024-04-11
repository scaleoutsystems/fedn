# MNIST (TensorFlow/Keras version)

This is a mimimalistic TF/Keras version of the Quickstart Tutorial (PyTorch). For more detailed explaination including a Jupyter Notebook with 
examples of API usage for starting and interacting with federated experiments, refer to that tutorial.

## Prerequisites
- [Python 3.8, 3.9, 3.10 or 3.11](https://www.python.org/downloads)
- [Docker](https://docs.docker.com/get-docker)
- [Docker Compose](https://docs.docker.com/compose/install)

## Running the example (pseudo-distributed)
Clone FEDn and locate into this directory.
```sh
git clone https://github.com/scaleoutsystems/fedn.git
cd fedn/examples/mnist-keras
```

### Build the compute package and the seed model (model to initalize the global model trail)

```sh
fedn package create --path client
```

```sh
fedn run build --path client
```

> You will now have two files,  `package.tgz` and `seed.npz`.

### Deploy FEDn 
Now we are ready to deploy FEDn and two clients with `docker-compose`.

```
docker-compose -f ../../docker-compose.yaml -f docker-compose.override.yaml up  
```

> **Note**: run with `--scale client=N` to start *N* clients.

### Run federated training 
Refer to this notebook to upload the package and seed model and run experiments:

 https://github.com/scaleoutsystems/fedn/blob/master/examples/mnist-pytorch/API_Example.ipynb 

## Clean up
You can clean up by running `docker-compose down`.
